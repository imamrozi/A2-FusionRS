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
import time
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
    save_results_yaml,
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
    # mode "mean" (default): skor per-aspek dirata-rata POLOS jadi 1 kolom
    # sentiment_score, drop-in pengganti SA global di SEMUA tempat (broad
    # propagation, termasuk sentiment_agg CBF) -- lihat score_dataframe().
    # mode "concat": skor per-aspek TANPA agregasi, dikirim sebagai vektor
    # mentah ke Fusion (bukan 1 skalar) -- lihat score_dataframe_per_aspect()
    # & tahap 7 di bawah. CBF tetap dapat 1 skor (rata-rata kolom aspek,
    # DITURUNKAN dari hasil ini, bukan panggilan predict_proba() terpisah).
    # mode "confidence_mean": skor per-aspek dirata-rata BERBOBOT confidence
    # jadi 1 kolom sentiment_score -- lihat score_dataframe_confidence_weighted().
    # Menguji apakah confidence-aware weighting memperbaiki kegagalan mode
    # "mean" polos (RMSE jauh lebih buruk dari SA global secara empiris).
    absa_mode = config.get("absa", {}).get("mode", "mean")
    aspect_names = list(scorer.aspect_keywords.keys())

    if absa_mode == "concat":
        # Cache BEDA NAMA dari mode mean, supaya keduanya bisa hidup
        # berdampingan di folder checkpoint yang sama tanpa saling menimpa.
        absa_cache = sa_checkpoint_dir / "absa_aspect_scores.csv"
        if absa_cache.exists():
            logger.info(
                "Cache skor ABSA-concat ditemukan di %s -- skip inference, langsung load.",
                absa_cache,
            )
            cache_df = pd.read_csv(absa_cache).set_index("review_id")
            for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                for aspect in aspect_names:
                    part_df[aspect] = part_df["review_id"].map(cache_df[aspect])
        else:
            for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                logger.info(
                    "=== ABSA-concat: menghitung skor per-aspek untuk split '%s' (%d baris) ===",
                    name, len(part_df),
                )
                t0 = time.time()
                aspect_df = scorer.score_dataframe_per_aspect(part_df)
                for aspect in aspect_names:
                    part_df[aspect] = aspect_df[aspect].values
                logger.info(
                    "=== ABSA-concat: split '%s' selesai dalam %.1f menit ===",
                    name, (time.time() - t0) / 60,
                )
            logger.info("Skor ABSA-concat selesai dihitung untuk semua split.")

            pd.concat(
                [
                    train_df[["review_id"] + aspect_names],
                    val_df[["review_id"] + aspect_names],
                    test_df[["review_id"] + aspect_names],
                ]
            ).to_csv(absa_cache, index=False)
            logger.info("Skor ABSA-concat disimpan ke cache %s.", absa_cache)

        # sentiment_score turunan (rata-rata kolom aspek) khusus utk CBF's
        # sentiment_agg -- fusion (tahap 7) pakai kolom aspek mentah langsung,
        # BUKAN kolom turunan ini.
        for part_df in [train_df, val_df, test_df]:
            part_df["sentiment_score"] = part_df[aspect_names].mean(axis=1)
    elif absa_mode == "confidence_mean":
        # Cache BEDA NAMA dari mode mean/concat -- ketiganya bisa hidup
        # berdampingan tanpa saling menimpa.
        absa_cache = sa_checkpoint_dir / "absa_confidence_mean_scores.csv"
        if absa_cache.exists():
            logger.info(
                "Cache skor ABSA-confidence ditemukan di %s -- skip inference, langsung load.",
                absa_cache,
            )
            score_map = pd.read_csv(absa_cache).set_index("review_id")["sentiment_score"]
            train_df["sentiment_score"] = train_df["review_id"].map(score_map)
            val_df["sentiment_score"] = val_df["review_id"].map(score_map)
            test_df["sentiment_score"] = test_df["review_id"].map(score_map)
        else:
            for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                logger.info(
                    "=== ABSA-confidence: menghitung skor untuk split '%s' (%d baris) ===",
                    name, len(part_df),
                )
                t0 = time.time()
                part_df["sentiment_score"] = scorer.score_dataframe_confidence_weighted(part_df)
                logger.info(
                    "=== ABSA-confidence: split '%s' selesai dalam %.1f menit ===",
                    name, (time.time() - t0) / 60,
                )
            logger.info("Skor ABSA-confidence selesai dihitung untuk semua split.")

            pd.concat(
                [
                    train_df[["review_id", "sentiment_score"]],
                    val_df[["review_id", "sentiment_score"]],
                    test_df[["review_id", "sentiment_score"]],
                ]
            ).to_csv(absa_cache, index=False)
            logger.info("Skor ABSA-confidence disimpan ke cache %s.", absa_cache)
    else:
        # Cache ke file BEDA NAMA dari sentiment_scores.csv milik
        # run_baseline.py, supaya tidak bentrok/menimpa cache skor SA global.
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
            # Dipisah per-split (bukan 1 log block untuk train+val+test
            # sekaligus) supaya progress terlihat jelas -- ABSA butuh
            # beberapa kali lebih banyak panggilan ke BERT dibanding SA
            # global (1x per aspek yang match per baris, lihat log
            # "ABSA: ... panggilan teks" di bawah), jadi tahap ini bisa
            # signifikan lebih lama dari yang diperkirakan.
            for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                logger.info(
                    "=== ABSA: menghitung skor untuk split '%s' (%d baris) ===", name, len(part_df)
                )
                t0 = time.time()
                part_df["sentiment_score"] = scorer.score_dataframe(part_df)
                logger.info(
                    "=== ABSA: split '%s' selesai dalam %.1f menit ===", name, (time.time() - t0) / 60
                )
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

    logger.info("Menghitung prediksi 3 stream pada test set...")
    test_deepmf_preds = deepmf_trainer.predict(test_df, user2idx, item2idx, rating_scale)
    test_cbf_preds = cbf_predictor.predict(test_df, rating_scale)

    if absa_mode == "concat":
        # Vektor mentah k-kolom (bukan 1 skalar) -- NMFDecisionTreeFusion
        # sudah digeneralisasi terima input 2D (lihat fusion_nmf_dt.py).
        train_sentiment_scores = train_df[aspect_names].values
        test_sentiment_scores = test_df[aspect_names].values
    else:
        train_sentiment_scores = train_df["sentiment_score"].values
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

    # Dispatch nama file/model_name/notes per mode -- 3 mode sekarang hidup
    # berdampingan, TIDAK saling menimpa hasil satu sama lain.
    mode_meta = {
        "concat": (
            "absa_ablation_concat",
            "baseline_darraz_with_absa_concat_ablation",
            "Varian ablasi: sama persis pipeline baseline_reimpl (DeepMF+CBF+NMF/DT "
            "digeneralisasi terima input multi-kolom, tidak ada perubahan perilaku "
            "lain), sentiment_score global diganti VEKTOR skor per-aspek ABSA "
            "(TANPA agregasi/rata-rata) langsung ke fusion -- CBF tetap pakai 1 "
            "skor turunan (rata-rata kolom aspek) untuk sentiment_agg. Ranking "
            "metrics pakai candidate set terbatas ke item test set -- lihat "
            "run_baseline.py.",
        ),
        "confidence_mean": (
            "absa_ablation_confidence_mean",
            "baseline_darraz_with_absa_confidence_weighted_ablation",
            "Varian ablasi: sama persis pipeline baseline_reimpl (DeepMF+CBF+NMF/DT, "
            "tidak diubah), sentiment_score diganti RATA-RATA BERBOBOT CONFIDENCE "
            "antar skor ABSA per-aspek (bukan rata-rata polos) -- menguji apakah "
            "confidence-aware weighting (pendekatan dari paper IEEE penulis "
            "sebelumnya) memperbaiki kegagalan mode 'mean' polos. Confidence dari "
            "margin skor + jumlah kalimat bukti per aspek, lihat "
            "score_dataframe_confidence_weighted() di absa_bert.py. Ranking "
            "metrics pakai candidate set terbatas ke item test set -- lihat "
            "run_baseline.py.",
        ),
        "mean": (
            "absa_ablation",
            "baseline_darraz_with_absa_sentiment_ablation",
            "Varian ablasi: sama persis pipeline baseline_reimpl (DeepMF+CBF+"
            "NMF/DT, tidak diubah), sentiment_score diganti skor ABSA "
            "keyword-based (bukan SA global) di SEMUA tempat pipeline "
            "memakainya (termasuk sentiment_agg di CBF). Ranking metrics "
            "pakai candidate set terbatas ke item test set -- lihat "
            "run_baseline.py.",
        ),
    }
    results_prefix, model_name, notes = mode_meta.get(absa_mode, mode_meta["mean"])
    results_path = results_dir / f"{results_prefix}_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"

    results_summary = {
        "model_name": model_name,
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": {int(k): v for k, v in precision_k.items()},
        "recall_at_k": {int(k): v for k, v in recall_k.items()},
        "ndcg_at_k": {int(k): v for k, v in ndcg_k.items()},
        "aspect_coverage": coverage_report,
        "notes": notes,
    }
    save_results_yaml(results_path, results_summary, config=config)

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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed dari config -- utk protokol multi-seed. "
        "Split & cache skor ABSA/SA TETAP dipakai bersama, cuma tahap 5-8 yang "
        "bervariasi -- run seed tambahan jauh lebih cepat dari run pertama.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    run_pipeline(cfg)
