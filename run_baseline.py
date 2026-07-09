"""
run_baseline.py

Entry point orkestrasi Fase 1: reimplementasi baseline Darraz et al. pada
domain Yelp dengan protokol evaluasi yang ketat (held-out test set, tanpa
leakage). Menjalankan seluruh pipeline: load -> split -> preprocess ->
sentiment (global) -> DeepMF -> CBF clustering -> fusion NMF+DT -> evaluasi.

PERINGATAN: script ini adalah skeleton orkestrasi. Sebelum dijalankan pada
eksperimen sesungguhnya:
1. Unduh dataset Yelp riil ke data/raw/ (lihat configs/yelp_config.yaml)
2. Validasi asumsi label sentiment (lihat sentiment_bert.py) terhadap
   detail metodologi baseline paper
3. Jalankan dulu pada subset kecil (misal 5.000 baris) untuk memverifikasi
   seluruh pipeline berjalan tanpa error sebelum full run yang memakan
   compute budget Colab

Usage:
    python run_baseline.py --config configs/yelp_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baseline.cbf_clustering import CBFConfig, CBFPredictor
from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset
from src.baseline.fusion_nmf_dt import FusionConfig, NMFDecisionTreeFusion
from src.baseline.sentiment_bert import (
    GlobalSentimentBERT,
    SentimentBertConfig,
    derive_sentiment_label,
)
from src.data_loader import YelpDatasetLoader
from src.evaluation.metrics import compute_rmse_mae, precision_recall_ndcg_at_k, sanity_check_rmse
from src.preprocessing import TextPreprocessor
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> None:
    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    split_cfg = config["split"]

    np.random.seed(exp_cfg["seed"])

    # ---------- 1. Load data ----------
    logger.info("=== Tahap 1: Memuat data ===")
    loader = YelpDatasetLoader(data_cfg["raw_path"], domain=exp_cfg["domain"])
    df = loader.load()
    df = loader.filter_min_interactions(
        df,
        min_reviews_per_user=data_cfg["min_reviews_per_user"],
        min_reviews_per_item=data_cfg["min_reviews_per_item"],
    )
    stats = loader.compute_stats(df)
    logger.info("Statistik dataset: %s", stats)

    # ---------- 2. Split ----------
    logger.info("=== Tahap 2: Membuat/memuat split ===")
    split_output_dir = Path(split_cfg["output_dir"])
    if (split_output_dir / "train.csv").exists():
        logger.info("Split sudah ada, memuat dari %s", split_output_dir)
        splits = UserBasedSplitGenerator.load(split_output_dir)
    else:
        generator = UserBasedSplitGenerator(
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            seed=exp_cfg["seed"],
            cold_start_holdout=split_cfg["cold_start_holdout"],
        )
        splits = generator.split(df)
        generator.save(splits, split_output_dir)

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # ---------- 3. Preprocessing ----------
    logger.info("=== Tahap 3: Preprocessing teks ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    # ---------- 4. Sentiment label (derive dari rating, filter netral) ----------
    logger.info("=== Tahap 4: Derivasi label sentiment & training SA global ===")
    for name, part in [("train", train_df), ("val", val_df)]:
        before = len(part)
        part = part[part["stars"] != 3].copy()
        part["sentiment_label"] = part["stars"].apply(derive_sentiment_label)
        logger.info(
            "%s: %d baris dibuang (rating netral=3) dari %d total", name, before - len(part), before
        )
        if name == "train":
            train_df_sa = part
        else:
            val_df_sa = part

    sa_config = SentimentBertConfig(
        model_name=config["sentiment_baseline"]["model_name"],
        max_length=data_cfg.get("max_seq_length", 512),
        batch_size=config["sentiment_baseline"]["batch_size"],
        learning_rate=config["sentiment_baseline"]["learning_rate"],
        epochs=config["sentiment_baseline"]["epochs"],
    )
    sa_model = GlobalSentimentBERT(sa_config)
    sa_model.fit(train_df_sa, val_df_sa)

    # Skor sentimen (probabilitas) untuk SEMUA baris (train/val/test),
    # dipakai sebagai fitur numerik ke fusion layer -- bukan label biner.
    train_df["sentiment_score"] = sa_model.predict_proba(train_df["text_bert"].tolist())
    val_df["sentiment_score"] = sa_model.predict_proba(val_df["text_bert"].tolist())
    test_df["sentiment_score"] = sa_model.predict_proba(test_df["text_bert"].tolist())

    # ---------- 5. DeepMF ----------
    logger.info("=== Tahap 5: Training DeepMF ===")
    all_users = pd.concat([train_df["user_id"], val_df["user_id"], test_df["user_id"]]).unique()
    all_items = pd.concat(
        [train_df["business_id"], val_df["business_id"], test_df["business_id"]]
    ).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}

    deepmf_config = DeepMFConfig(
        embedding_dim=config["deepmf"]["embedding_dim"],
        hidden_layers=tuple(config["deepmf"]["hidden_layers"]),
        dropout=config["deepmf"]["dropout"],
        batch_size=config["deepmf"]["batch_size"],
        learning_rate=config["deepmf"]["learning_rate"],
        negative_sampling_ratio=config["deepmf"]["negative_sampling_ratio"],
    )

    train_interactions = InteractionDataset(
        train_df, user2idx, item2idx, len(all_items), deepmf_config.negative_sampling_ratio
    )
    val_interactions = InteractionDataset(
        val_df, user2idx, item2idx, len(all_items), negative_ratio=0
    )

    deepmf_trainer = DeepMFTrainer(len(all_users), len(all_items), deepmf_config)
    deepmf_trainer.fit(train_interactions, val_interactions)

    # ---------- 6. CBF Clustering ----------
    logger.info("=== Tahap 6: Content-Based Filtering & Clustering ===")
    full_df_for_items = pd.concat([train_df, val_df, test_df], ignore_index=True)

    cbf_config = CBFConfig(
        method=config["cbf_clustering"]["method"],
        k_min=2,
        k_max=20,
    )
    cbf_predictor = CBFPredictor(cbf_config=cbf_config)
    cbf_predictor.fit(full_df_for_items, train_df)

    logger.info(
        "CBF clustering selesai: K optimal=%d (metode=%s)",
        cbf_predictor.clusterer.best_k,
        cbf_config.method,
    )

    # ---------- 7. Fusion NMF + DecisionTree ----------
    logger.info("=== Tahap 7: Fusion NMF + DecisionTreeRegressor ===")

    rating_scale = (1.0, 5.0)

    logger.info("Menghitung prediksi 3 stream pada train set...")
    train_deepmf_preds = deepmf_trainer.predict(train_df, user2idx, item2idx, rating_scale)
    train_cbf_preds = cbf_predictor.predict(train_df, rating_scale)
    train_sentiment_scores = train_df["sentiment_score"].values

    logger.info("Menghitung prediksi 3 stream pada test set...")
    test_deepmf_preds = deepmf_trainer.predict(test_df, user2idx, item2idx, rating_scale)
    test_cbf_preds = cbf_predictor.predict(test_df, rating_scale)
    test_sentiment_scores = test_df["sentiment_score"].values

    fusion_config = FusionConfig(
        nmf_components=config["fusion_baseline"]["nmf_components"],
        dt_max_depth=config["fusion_baseline"]["dt_max_depth"],
    )
    fusion_model = NMFDecisionTreeFusion(fusion_config)
    fusion_model.fit(
        sentiment_scores=train_sentiment_scores,
        deepmf_preds=train_deepmf_preds,
        cbf_preds=train_cbf_preds,
        y_true_ratings=train_df["stars"].values,
    )

    test_final_preds = fusion_model.predict(
        sentiment_scores=test_sentiment_scores,
        deepmf_preds=test_deepmf_preds,
        cbf_preds=test_cbf_preds,
    )
    test_final_preds = np.clip(test_final_preds, rating_scale[0], rating_scale[1])

    # ---------- 8. Evaluasi akhir ----------
    logger.info("=== Tahap 8: Evaluasi akhir (pada test set held-out) ===")

    y_true = test_df["stars"].values
    rmse, mae = compute_rmse_mae(y_true, test_final_preds)
    sanity_check_rmse(rmse, rating_scale)

    logger.info("=" * 60)
    logger.info("HASIL BASELINE REIMPLEMENTATION (domain: %s)", exp_cfg["domain"])
    logger.info("RMSE: %.4f", rmse)
    logger.info("MAE : %.4f", mae)
    logger.info("=" * 60)

    # Ranking metrics (Precision/Recall/NDCG@K) -- SIMPLIFIKASI YANG PERLU
    # DICATAT: candidate set per user dibatasi hanya pada item yang muncul
    # di test set (bukan seluruh katalog item), karena ranking terhadap
    # seluruh katalog jutaan item tidak feasible dihitung untuk setiap user
    # pada tahap baseline ini. Ini praktik umum evaluasi offline RS skala
    # menengah, TAPI harus dinyatakan eksplisit sebagai batasan di bagian
    # metodologi/limitasi manuskrip -- angka Precision/Recall/NDCG absolut
    # tidak boleh dibandingkan langsung dengan studi lain yang memakai
    # protokol full-catalog ranking.
    logger.info("Menghitung ranking metrics (Precision/Recall/NDCG@K)...")

    test_df_eval = test_df.copy()
    test_df_eval["pred_score"] = test_final_preds

    relevance_threshold = 4.0  # rating >=4 dianggap "relevant", konsisten dgn literatur umum
    ranked_items_per_user: dict = {}
    relevant_items_per_user: dict = {}

    for user_id, group in test_df_eval.groupby("user_id"):
        ranked = group.sort_values("pred_score", ascending=False)["business_id"].tolist()
        ranked_items_per_user[user_id] = ranked
        relevant = set(group[group["stars"] >= relevance_threshold]["business_id"])
        relevant_items_per_user[user_id] = relevant

    k_values = config["evaluation"]["k_values"]
    precision_k, recall_k, ndcg_k = precision_recall_ndcg_at_k(
        ranked_items_per_user, relevant_items_per_user, k_values
    )

    logger.info("Precision@K: %s", precision_k)
    logger.info("Recall@K   : %s", recall_k)
    logger.info("NDCG@K     : %s", ndcg_k)

    # Simpan hasil ke file untuk dibandingkan dengan model lain (SVD, NCF,
    # DeepFM, A2-FusionRS, dan varian ablasi) yang akan dijalankan terpisah.
    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"baseline_reimpl_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"

    results_summary = {
        "model_name": "baseline_reimplementation_darraz_et_al",
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": {int(k): v for k, v in precision_k.items()},
        "recall_at_k": {int(k): v for k, v in recall_k.items()},
        "ndcg_at_k": {int(k): v for k, v in ndcg_k.items()},
        "notes": (
            "Ranking metrics dihitung dengan candidate set terbatas pada item "
            "test set (bukan full-catalog) -- lihat komentar di run_baseline.py"
        ),
    }
    with open(results_path, "w") as f:
        yaml.safe_dump(results_summary, f, allow_unicode=True)
    logger.info("Hasil evaluasi disimpan ke %s", results_path)

    logger.info(
        "Pipeline baseline reimplementation SELESAI. Bandingkan RMSE=%.4f "
        "ini dengan angka yang dilaporkan baseline paper (0.01-0.02) -- jika "
        "jauh berbeda, dokumentasikan sebagai temuan metodologis di bagian "
        "Discussion (lihat diskusi anomali RMSE sebelumnya).",
        rmse,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan pipeline reimplementasi baseline")
    parser.add_argument("--config", type=str, default="configs/yelp_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)
