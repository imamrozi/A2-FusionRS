"""
scripts/diagnose_deepmf_collapse.py

Investigasi cepat (Temuan A3 lanjutan, reports/methodology_audit_2026-07-26.md):
seberapa SERING DeepMF (config default, SGD lr=0,001) kolaps ke prediktor
nyaris-konstan (std~0) -- ditemukan pertama di seed 456 (RMSE 3,14 sebelum
fix B1 / 1,21 setelahnya), TERNYATA JUGA muncul di seed 42 setelah fix B1
diterapkan (smoke test A1, test_deepmf_preds std=0,0000).

DeepMF TIDAK butuh teks review sama sekali (InteractionDataset cuma pakai
user_id/business_id/stars) -- script ini SENGAJA skip preprocessing/ABSA/
CBF/fusion, cuma load split mentah (CSV langsung, TANPA TextPreprocessor)
+ latih DeepMF murni per seed, supaya jauh lebih cepat drpd run pipeline
penuh (~2-3 menit/seed vs ~15-18 menit/run pipeline).

Usage:
    python scripts/diagnose_deepmf_collapse.py --config configs/tripadvisor_hotel_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset
from src.config_utils import load_config
from src.split_generator import UserBasedSplitGenerator

RATING_SCALE = (1.0, 5.0)
SEEDS = [42, 123, 456, 789, 1011]


def run_one_seed(seed, train_df, val_df, test_df, user2idx, item2idx, n_items, deepmf_kwargs):
    torch.manual_seed(seed)
    config = DeepMFConfig(
        embedding_dim=deepmf_kwargs["embedding_dim"],
        hidden_layers=tuple(deepmf_kwargs["hidden_layers"]),
        dropout=deepmf_kwargs["dropout"],
        batch_size=deepmf_kwargs["batch_size"],
        learning_rate=deepmf_kwargs["learning_rate"],
        negative_sampling_ratio=deepmf_kwargs["negative_sampling_ratio"],
        optimizer=deepmf_kwargs.get("optimizer", "sgd"),
        weight_decay=deepmf_kwargs.get("weight_decay", 0.0),
        epochs=deepmf_kwargs.get("epochs", 20),
    )

    train_interactions = InteractionDataset(
        train_df, user2idx, item2idx, n_items, config.negative_sampling_ratio, seed=seed,
    )
    val_interactions = InteractionDataset(
        val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=seed,
    )

    torch.manual_seed(seed)  # reseed tepat sebelum training (fix B1)
    trainer = DeepMFTrainer(len(user2idx), n_items, config)
    trainer.fit(train_interactions, val_interactions)

    test_preds = trainer.predict(test_df, user2idx, item2idx, RATING_SCALE)
    train_preds_sample = trainer.predict(train_df.sample(min(5000, len(train_df)), random_state=seed), user2idx, item2idx, RATING_SCALE)

    return {
        "seed": seed,
        "test_std": float(np.std(test_preds)),
        "test_mean": float(np.mean(test_preds)),
        "test_n_unique": int(len(np.unique(test_preds))),
        "train_sample_std": float(np.std(train_preds_sample)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/tripadvisor_hotel_config.yaml")
    parser.add_argument("--optimizer", type=str, default=None, help="Override optimizer (sgd/adam/adamw).")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning_rate.")
    args = parser.parse_args()

    config = load_config(args.config)
    split_cfg = config["split"]

    splits = UserBasedSplitGenerator.load(Path(split_cfg["output_dir"]))
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    all_users = pd.concat([train_df["user_id"], val_df["user_id"], test_df["user_id"]]).unique()
    all_items = pd.concat([train_df["business_id"], val_df["business_id"], test_df["business_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}
    n_items = len(all_items)

    deepmf_kwargs = dict(config["deepmf"])
    if args.optimizer is not None:
        deepmf_kwargs["optimizer"] = args.optimizer
    if args.learning_rate is not None:
        deepmf_kwargs["learning_rate"] = args.learning_rate

    print(f"=== Domain: {config['experiment']['domain']} -- {len(SEEDS)} seed ===")
    results = []
    for seed in SEEDS:
        r = run_one_seed(seed, train_df, val_df, test_df, user2idx, item2idx, n_items, deepmf_kwargs)
        collapsed = "KOLAPS" if r["test_std"] < 1e-3 else "sehat"
        print(
            f"seed={r['seed']:5d}  test_std={r['test_std']:.4f}  test_mean={r['test_mean']:.4f}  "
            f"test_n_unique={r['test_n_unique']:5d}/{len(test_df)}  train_sample_std={r['train_sample_std']:.4f}  [{collapsed}]"
        )
        results.append(r)

    n_collapsed = sum(1 for r in results if r["test_std"] < 1e-3)
    print(f"\n=== RINGKASAN: {n_collapsed}/{len(SEEDS)} seed KOLAPS (test_std < 0,001) ===")


if __name__ == "__main__":
    main()
