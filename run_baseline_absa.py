"""
run_baseline_absa.py

Varian ablasi Fase 1: pipeline baseline Darraz et al. yang SAMA PERSIS
(load -> split -> preprocess -> DeepMF -> CBF -> fusion NMF/DT -> evaluasi),
TAPI stream sentimen GLOBAL (satu skor BERT per review) diganti skor
ber-ASPEK (`src/a2fusionrs/absa_bert.py`, keyword-based, reuse model SA yang
SUDAH di-checkpoint dari run_baseline.py -- TIDAK ADA training BERT baru di
sini).

Ini duplikat run_baseline.py, HANYA tahap 4 yang beda -- tahap 1,3,5,6,7,8
sengaja disalin verbatim (bukan di-refactor jadi pluggable) untuk meminimalkan
risiko terhadap pipeline baseline yang sudah divalidasi & diperbaiki
berkali-kali. Tahap 2 juga sedikit beda: split WAJIB sudah ada (load-only,
tidak auto-generate), sama seperti run_classical_cf.py.

Skor ABSA menggantikan kolom `sentiment_score` di SEMUA tempat pipeline
memakainya (broad propagation) -- termasuk fitur `sentiment_agg` di CBF,
bukan cuma input fusion. Ini keputusan desain yang disengaja (lihat
plan/diskusi terkait): implementasi paling sederhana, ZERO perubahan ke
cbf_clustering.py/fusion_nmf_dt.py, karena keduanya cuma baca nama kolom
tanpa peduli asal skornya.

PRASYARAT: jalankan run_baseline.py (config domain yang sama) SEKALI dulu
sampai tahap 4 selesai, supaya checkpoint model SA (`sentiment_bert/`) ada.
Script ini akan berhenti dengan error jelas kalau checkpoint belum ada --
SENGAJA tidak fallback training baru.

Usage:
    python run_baseline_absa.py --config configs/yelp_config_absa.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.a2fusionrs.absa_bert import ABSAConfig, KeywordAspectSentimentScorer
from src.baseline.cbf_clustering import CBFConfig, CBFPredictor
from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset
from src.baseline.fusion_nmf_dt import FusionConfig, NMFDecisionTreeFusion
from src.baseline.sentiment_bert import GlobalSentimentBERT, SentimentBertConfig
from src.evaluation.metrics import (
    compute_rmse_mae,
    precision_recall_ndcg_at_k,
    sanity_check_rmse,
    save_predictions,
)
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
    torch.manual_seed(exp_cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(exp_cfg["seed"])

    # ---------- 1. Load data (TIDAK dipakai langsung -- split sudah final) --
    # Tahap 1 baseline (load raw CSV) sengaja DILEWATI di sini: kita hanya
    # perlu split yang sudah tersimpan, bukan raw dataset lagi.

    # ---------- 2. Split -- WAJIB sudah ada (load-only, sama spirit dengan
    # run_classical_cf.py: JANGAN generate split baru untuk arm ini) ----------
    logger.info("=== Tahap 2: Memuat split (WAJIB sudah ada) ===")
    split_output_dir = Path(split_cfg["output_dir"])
    splits = UserBasedSplitGenerator.load(split_output_dir)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    # ---------- 3. Preprocessing ----------
    logger.info("=== Tahap 3: Preprocessing teks ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    # ---------- 4. ABSA (ganti SA global) ----------
    logger.info("=== Tahap 4 (ABSA): Skor sentimen ber-aspek (ganti SA global) ===")

    sa_config = SentimentBertConfig(
        model_name=config["sentiment_baseline"]["model_name"],
        max_length=data_cfg.get("max_seq_length", 512),
        batch_size=config["sentiment_baseline"]["batch_size"],
        learning_rate=config["sentiment_baseline"]["learning_rate"],
        epochs=config["sentiment_baseline"]["epochs"],
    )

    # WAJIB checkpoint SA sudah ada dari run_baseline.py -- TIDAK training baru.
    sa_checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / "sentiment_bert"
    if not (sa_checkpoint_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Checkpoint model SA tidak ditemukan di {sa_checkpoint_dir}. "
            "run_baseline_absa.py TIDAK melakukan training BERT baru -- "
            "jalankan run_baseline.py (config domain yang sama, logging."
            "checkpoint_dir yang sama) sampai tahap 4 selesai terlebih dahulu."
        )
    logger.info("Memuat model SA dari checkpoint %s (dipakai ulang, tanpa training baru).", sa_checkpoint_dir)
    sa_model = GlobalSentimentBERT.load(str(sa_checkpoint_dir), sa_config)

    absa_config = ABSAConfig(
        domain=exp_cfg["domain"],
        aggregation=config.get("absa", {}).get("aggregation", "mean"),
    )
    scorer = KeywordAspectSentimentScorer(sa_model, absa_config)

    # Cache ke file BEDA NAMA dari sentiment_scores.csv milik run_baseline.py,
    # supaya tidak bentrok/menimpa cache skor SA global di folder yang sama.
    absa_scores_cache = sa_checkpoint_dir / "absa_sentiment_scores.csv"
    if absa_scores_cache.exists():
        logger.info(
            "Cache skor ABSA ditemukan di %s -- skip inference, langsung load.",
            absa_scores_cache,
        )
        score_map = pd.read_csv(absa_scores_cache).set_index("review_id")["sentiment_score"]
        train_df["sentiment_score"] = train_df["review_id"].map(score_map)
        val_df["sentiment_score"] = val_df["review_id"].map(score_map)
        test_df["sentiment_score"] = test_df["review_id"].map(score_map)
    else:
        logger.info(
            "Menghitung skor ABSA untuk %d baris train + %d val + %d test...",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        train_df["sentiment_score"] = scorer.score_dataframe(train_df)
        val_df["sentiment_score"] = scorer.score_dataframe(val_df)
        test_df["sentiment_score"] = scorer.score_dataframe(test_df)
        logger.info("Skor ABSA selesai dihitung untuk semua split.")

        pd.concat(
            [
                train_df[["review_id", "sentiment_score"]],
                val_df[["review_id", "sentiment_score"]],
                test_df[["review_id", "sentiment_score"]],
            ]
        ).to_csv(absa_scores_cache, index=False)
        logger.info("Skor ABSA disimpan ke cache %s.", absa_scores_cache)

    # Diagnostik cakupan aspek -- WAJIB dilog & disimpan (bukan nice-to-have),
    # cakupan rendah = ablasi kurang bermakna (skor konvergen ke fallback lagi).
    coverage_report = scorer.aspect_coverage_report(pd.concat([train_df, val_df, test_df], ignore_index=True))
    logger.info("Cakupan aspek ABSA: %s", coverage_report)

    # ---------- 5. DeepMF (verbatim sama dengan run_baseline.py) ----------
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
        train_df,
        user2idx,
        item2idx,
        len(all_items),
        deepmf_config.negative_sampling_ratio,
        seed=exp_cfg["seed"],
    )
    val_interactions = InteractionDataset(
        val_df, user2idx, item2idx, len(all_items), negative_ratio=0, seed=exp_cfg["seed"]
    )

    deepmf_trainer = DeepMFTrainer(len(all_users), len(all_items), deepmf_config)
    deepmf_trainer.fit(train_interactions, val_interactions)

    # ---------- 6. CBF Clustering (verbatim -- otomatis pakai sentiment_score
    # ABSA karena build_item_dataframe() baca kolom itu apa adanya) ----------
    logger.info("=== Tahap 6: Content-Based Filtering & Clustering ===")
    full_df_for_items = pd.concat([train_df, val_df, test_df], ignore_index=True)

    cbf_config = CBFConfig(
        method=config["cbf_clustering"]["method"],
        k_min=2,
        k_max=20,
        pca_components=config["cbf_clustering"].get("pca_components", 50),
        random_state=exp_cfg["seed"],
    )
    cbf_predictor = CBFPredictor(cbf_config=cbf_config)
    cbf_predictor.fit(full_df_for_items, train_df)

    logger.info(
        "CBF clustering selesai: K optimal=%d (metode=%s)",
        cbf_predictor.clusterer.best_k,
        cbf_config.method,
    )

    # ---------- 7. Fusion NMF + DecisionTree (verbatim) ----------
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
        random_state=exp_cfg["seed"],
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
    logger.info("HASIL ABLASI ABSA (domain: %s)", exp_cfg["domain"])
    logger.info("RMSE: %.4f", rmse)
    logger.info("MAE : %.4f", mae)
    logger.info("=" * 60)

    logger.info("Menghitung ranking metrics (Precision/Recall/NDCG@K)...")
    test_df_eval = test_df.copy()
    test_df_eval["pred_score"] = test_final_preds

    relevance_threshold = 4.0
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

    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"absa_ablation_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"

    results_summary = {
        "model_name": "baseline_darraz_with_absa_sentiment_ablation",
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": {int(k): v for k, v in precision_k.items()},
        "recall_at_k": {int(k): v for k, v in recall_k.items()},
        "ndcg_at_k": {int(k): v for k, v in ndcg_k.items()},
        "aspect_coverage": coverage_report,
        "notes": (
            "Varian ablasi: sama persis pipeline baseline_reimpl (DeepMF+CBF+"
            "NMF/DT, tidak diubah), sentiment_score diganti skor ABSA "
            "keyword-based (bukan SA global) di SEMUA tempat pipeline "
            "memakainya (termasuk sentiment_agg di CBF). Ranking metrics "
            "pakai candidate set terbatas ke item test set -- lihat "
            "run_baseline.py."
        ),
    }
    with open(results_path, "w") as f:
        yaml.safe_dump(results_summary, f, allow_unicode=True)
    logger.info("Hasil evaluasi ABSA disimpan ke %s", results_path)

    predictions_path = results_dir / f"predictions_{results_path.stem}.csv"
    save_predictions(predictions_path, test_df, test_final_preds)

    logger.info(
        "Pipeline ablasi ABSA SELESAI. RMSE=%.4f -- bandingkan dengan "
        "baseline_reimpl_%s_seed%d.yaml (SA global) untuk menilai efek "
        "aspect-awareness pada sentiment_score.",
        rmse,
        exp_cfg["domain"],
        exp_cfg["seed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan varian ablasi ABSA dari pipeline baseline")
    parser.add_argument("--config", type=str, default="configs/yelp_config_absa.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg)
