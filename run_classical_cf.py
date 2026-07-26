"""
run_classical_cf.py

Entry point orkestrasi untuk baseline collaborative-filtering RINGAN
(Item-KNN, SVD) -- pembanding kompleksitas/akurasi terhadap DeepMF, TIDAK
terintegrasi ke pipeline fusion baseline hybrid (run_baseline.py). Alur:
load split (SAMA PERSIS dengan run_baseline.py) -> fit -> predict -> evaluasi
(modul evaluasi identik dengan run_baseline.py) -> simpan hasil.

TIDAK ada preprocessing/BERT/CBF/fusion di sini -- murni CF dari rating
1-5 (user_id, business_id, stars).

WAJIB split sudah ada (hasil run_baseline.py) -- script ini SENGAJA TIDAK
generate split baru sendiri, supaya invariant "semua model dibandingkan pada
split identik" dipaksa secara struktural (bukan cuma dokumentasi).

Usage:
    python run_classical_cf.py --config configs/yelp_config.yaml
    python run_classical_cf.py --config configs/yelp_config.yaml --algorithm item_knn
    python run_classical_cf.py --config configs/yelp_config.yaml --algorithm svd
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.baseline.classical_cf import ClassicalCFConfig, ClassicalCFTrainer
from src.config_utils import load_config
from src.evaluation.metrics import (
    compute_rmse_mae,
    sanity_check_rmse,
    save_predictions,
    save_results_yaml,
)
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_one_algorithm(algorithm: str, config: dict, train_df, test_df) -> None:
    exp_cfg = config["experiment"]
    cf_cfg = config["classical_cf"][algorithm]
    rating_scale = (1.0, 5.0)

    logger.info("=== Melatih %s ===", algorithm)
    if algorithm == "item_knn":
        cfg = ClassicalCFConfig(
            algorithm="item_knn",
            knn_k=cf_cfg["knn_k"],
            knn_min_k=cf_cfg["knn_min_k"],
            knn_sim_name=cf_cfg["knn_sim_name"],
            knn_user_based=cf_cfg["knn_user_based"],
            random_state=exp_cfg["seed"],
        )
    elif algorithm == "svd":
        cfg = ClassicalCFConfig(
            algorithm="svd",
            svd_n_factors=cf_cfg["svd_n_factors"],
            svd_n_epochs=cf_cfg["svd_n_epochs"],
            svd_lr_all=cf_cfg["svd_lr_all"],
            svd_reg_all=cf_cfg["svd_reg_all"],
            random_state=exp_cfg["seed"],
        )
    else:
        raise ValueError(f"algorithm '{algorithm}' tidak dikenal -- gunakan 'item_knn' atau 'svd'")

    trainer = ClassicalCFTrainer(cfg)
    trainer.fit(train_df, rating_scale)
    test_preds = trainer.predict(test_df, rating_scale)

    # ---------- Evaluasi (identik dengan run_baseline.py tahap 8) ----------
    y_true = test_df["stars"].values
    rmse, mae = compute_rmse_mae(y_true, test_preds)
    sanity_check_rmse(rmse, rating_scale)

    logger.info("=" * 60)
    logger.info("HASIL %s (domain: %s)", algorithm.upper(), exp_cfg["domain"])
    logger.info("RMSE: %.4f", rmse)
    logger.info("MAE : %.4f", mae)
    logger.info("=" * 60)

    # Ranking metrics (Precision/Recall/NDCG@K) DINONAKTIFKAN (Temuan A1,
    # reports/methodology_audit_2026-07-26.md) -- lihat komentar identik di
    # run_baseline.py utk alasan lengkap.

    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = (
        results_dir / f"classical_cf_{algorithm}_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"
    )
    results_summary = {
        "model_name": f"classical_cf_{algorithm}",
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "notes": (
            "Baseline CF ringan (murni CF, tanpa sentiment/content-based), "
            "TIDAK bagian dari reimplementasi hybrid Darraz et al. Ranking "
            "metrics (Precision/Recall/NDCG@K) SENGAJA TIDAK dihitung/"
            "dilaporkan (Temuan A1, reports/methodology_audit_2026-07-26.md)."
        ),
    }
    save_results_yaml(results_path, results_summary, config=config)

    predictions_path = results_dir / f"predictions_{results_path.stem}.csv"
    save_predictions(predictions_path, test_df, test_preds)


def run_pipeline(config: dict, algorithms: list[str] | None = None) -> None:
    exp_cfg = config["experiment"]
    split_cfg = config["split"]

    np.random.seed(exp_cfg["seed"])

    split_output_dir = Path(split_cfg["output_dir"])
    logger.info("Memuat split dari %s (WAJIB sudah ada, lihat run_baseline.py)", split_output_dir)
    splits = UserBasedSplitGenerator.load(split_output_dir)
    train_df, test_df = splits["train"], splits["test"]

    algorithms = algorithms or config["classical_cf"]["algorithms"]
    for algorithm in algorithms:
        run_one_algorithm(algorithm, config, train_df, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan baseline CF ringan (Item-KNN, SVD)")
    parser.add_argument("--config", type=str, default="configs/yelp_config.yaml")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["item_knn", "svd"],
        default=None,
        help="Jalankan satu algoritma saja (default: semua yang ada di config.classical_cf.algorithms)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed dari config -- utk protokol multi-seed.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    algos = [args.algorithm] if args.algorithm else None
    run_pipeline(cfg, algorithms=algos)
