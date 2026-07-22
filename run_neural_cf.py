"""
run_neural_cf.py

Entry point orkestrasi untuk baseline collaborative-filtering NEURAL EKSTERNAL
(NeuMF, DeepFM) -- pembanding state-of-the-art terhadap A2-FusionRS yang
diminta reviewer Q1. Alur IDENTIK dengan run_classical_cf.py: load split
(SAMA PERSIS dengan semua model) -> fit -> predict -> evaluasi (modul evaluasi
identik) -> simpan hasil (YAML + predictions CSV berbasis review_id, kompatibel
run_significance_test.py).

MURNI CF dari rating 1-5 (user_id, business_id, stars); TANPA preprocessing/
BERT/CBF/fusion. WAJIB split sudah ada (hasil run_baseline.py) -- script ini
SENGAJA TIDAK generate split baru, supaya invariant "semua model dibandingkan
pada split identik" dipaksa secara struktural.

Hyperparameter neural_cf boleh diletakkan di config (blok `neural_cf:`), tapi
kalau TIDAK ADA, dipakai default wajar built-in (NeuralCFConfig) -- supaya
tidak perlu mengedit belasan file config domain yang sudah ada.

Usage:
    python run_neural_cf.py --config configs/yelp_config.yaml --model neumf
    python run_neural_cf.py --config configs/yelp_config.yaml --model deepfm --seed 123
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.baseline.neural_cf import NeuralCFConfig, NeuralCFTrainer
from src.config_utils import load_config
from src.evaluation.metrics import (
    compute_rmse_mae,
    precision_recall_ndcg_at_k,
    sanity_check_rmse,
    save_predictions,
    save_results_yaml,
)
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _build_config(model: str, config: dict, seed: int) -> NeuralCFConfig:
    """Ambil hyperparameter dari config.neural_cf[model] kalau ada, kalau
    tidak pakai default NeuralCFConfig (dengan seed dari experiment)."""
    base = NeuralCFConfig(model=model, seed=seed)
    nc = config.get("neural_cf", {}).get(model, {}) if isinstance(config.get("neural_cf"), dict) else {}
    for key in ("embedding_dim", "dropout", "batch_size", "learning_rate", "weight_decay", "epochs"):
        if key in nc:
            setattr(base, key, nc[key])
    if "mlp_layers" in nc:
        base.mlp_layers = tuple(nc["mlp_layers"])
    return base


def run_one_model(model: str, config: dict, train_df, val_df, test_df) -> None:
    exp_cfg = config["experiment"]
    rating_scale = (1.0, 5.0)

    logger.info("=== Melatih %s ===", model)
    cfg = _build_config(model, config, exp_cfg["seed"])
    n_users = train_df["user_id"].nunique()
    n_items = train_df["business_id"].nunique()
    trainer = NeuralCFTrainer(n_users, n_items, cfg)
    t0 = time.time()
    trainer.fit(train_df, val_df, rating_scale)
    train_time = time.time() - t0
    t0 = time.time()
    test_preds = trainer.predict(test_df)
    predict_time = time.time() - t0
    n_params = int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))

    # ---------- Evaluasi (identik dengan run_baseline.py tahap 8) ----------
    y_true = test_df["stars"].values
    rmse, mae = compute_rmse_mae(y_true, test_preds)
    sanity_check_rmse(rmse, rating_scale)

    logger.info("=" * 60)
    logger.info("HASIL %s (domain: %s, seed: %d)", model.upper(), exp_cfg["domain"], exp_cfg["seed"])
    logger.info("RMSE: %.4f | MAE: %.4f", rmse, mae)
    logger.info("=" * 60)

    test_df_eval = test_df.copy()
    test_df_eval["pred_score"] = test_preds
    relevance_threshold = 4.0
    ranked_items_per_user: dict = {}
    relevant_items_per_user: dict = {}
    for user_id, group in test_df_eval.groupby("user_id"):
        ranked_items_per_user[user_id] = group.sort_values("pred_score", ascending=False)["business_id"].tolist()
        relevant_items_per_user[user_id] = set(group[group["stars"] >= relevance_threshold]["business_id"])

    k_values = config["evaluation"]["k_values"]
    precision_k, recall_k, ndcg_k = precision_recall_ndcg_at_k(
        ranked_items_per_user, relevant_items_per_user, k_values
    )
    logger.info("Precision@K: %s | Recall@K: %s | NDCG@K: %s", precision_k, recall_k, ndcg_k)

    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"neural_cf_{model}_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"
    results_summary = {
        "model_name": f"neural_cf_{model}",
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": {int(k): v for k, v in precision_k.items()},
        "recall_at_k": {int(k): v for k, v in recall_k.items()},
        "ndcg_at_k": {int(k): v for k, v in ndcg_k.items()},
        "embedding_dim": cfg.embedding_dim,
        "mlp_layers": list(cfg.mlp_layers),
        "epochs": cfg.epochs,
        "train_time_seconds": train_time,
        "predict_time_seconds": predict_time,
        "n_parameters": n_params,
        "notes": (
            "Baseline CF neural eksternal (NeuMF/DeepFM diadaptasi ke regresi "
            "rating), pembanding state-of-the-art A2-FusionRS. Split & protokol "
            "evaluasi IDENTIK dengan model lain; ranking metrics pakai candidate "
            "set terbatas ke item test set (lihat run_baseline.py)."
        ),
    }
    save_results_yaml(results_path, results_summary, config=config)

    predictions_path = results_dir / f"predictions_{results_path.stem}.csv"
    save_predictions(predictions_path, test_df, test_preds)


def run_pipeline(config: dict, models: list[str] | None = None) -> None:
    exp_cfg = config["experiment"]
    split_cfg = config["split"]

    np.random.seed(exp_cfg["seed"])
    torch.manual_seed(exp_cfg["seed"])

    split_output_dir = Path(split_cfg["output_dir"])
    logger.info("Memuat split dari %s (WAJIB sudah ada, lihat run_baseline.py)", split_output_dir)
    splits = UserBasedSplitGenerator.load(split_output_dir)
    train_df, test_df = splits["train"], splits["test"]
    val_df = splits.get("val")

    for model in (models or ["neumf", "deepfm"]):
        run_one_model(model, config, train_df, val_df, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan baseline CF neural eksternal (NeuMF, DeepFM)")
    parser.add_argument("--config", type=str, default="configs/yelp_config.yaml")
    parser.add_argument("--model", type=str, choices=["neumf", "deepfm"], default=None,
                        help="Jalankan satu model saja (default: neumf + deepfm)")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed (protokol multi-seed)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    models = [args.model] if args.model else None
    run_pipeline(cfg, models=models)
