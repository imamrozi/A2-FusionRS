"""
tests/test_deepmf_oof.py

Regresi utk compute_oof_predictions() (src/baseline/deepmf.py) -- koreksi
out-of-fold DeepMF utk train_deepmf_preds, hindari leakage in-sample
klasik stacked model (lihat docstring fungsi itu utk motivasi lengkap).

Epoch/embedding_dim SENGAJA kecil supaya test cepat -- properti yg diuji
struktural (bentuk output, tidak ada NaN, dalam rating_scale), BUKAN
akurasi model (itu diukur via smoke test end-to-end run_baseline*.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.deepmf import (  # noqa: E402
    DeepMFConfig,
    InteractionDataset,
    compute_oof_predictions,
)

RATING_SCALE = (1.0, 5.0)


@pytest.fixture
def synthetic_interactions():
    rng = np.random.RandomState(42)
    n_users, n_items_n, n_rows = 15, 10, 80
    train_df = pd.DataFrame({
        "user_id": [f"u{rng.randint(0, n_users)}" for _ in range(n_rows)],
        "business_id": [f"i{rng.randint(0, n_items_n)}" for _ in range(n_rows)],
        "stars": rng.uniform(1.0, 5.0, size=n_rows).astype(np.float32),
    })
    val_df = pd.DataFrame({
        "user_id": [f"u{rng.randint(0, n_users)}" for _ in range(20)],
        "business_id": [f"i{rng.randint(0, n_items_n)}" for _ in range(20)],
        "stars": rng.uniform(1.0, 5.0, size=20).astype(np.float32),
    })
    all_users = sorted(set(train_df["user_id"]) | set(val_df["user_id"]))
    all_items = sorted(set(train_df["business_id"]) | set(val_df["business_id"]))
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}
    return train_df, val_df, user2idx, item2idx, len(all_items)


def test_compute_oof_predictions_shape_and_bounds(synthetic_interactions):
    train_df, val_df, user2idx, item2idx, n_items = synthetic_interactions
    config = DeepMFConfig(embedding_dim=8, hidden_layers=(16, 8), epochs=2, batch_size=32)
    val_dataset = InteractionDataset(val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=42)

    oof_preds = compute_oof_predictions(
        train_df, val_dataset, user2idx, item2idx, n_items, config,
        rating_scale=RATING_SCALE, seed=42, n_folds=3,
    )

    assert len(oof_preds) == len(train_df)
    assert np.isfinite(oof_preds).all()
    assert (oof_preds >= RATING_SCALE[0]).all() and (oof_preds <= RATING_SCALE[1]).all()


def test_compute_oof_predictions_deterministic_with_same_seed(synthetic_interactions):
    """Seed sama -> hasil OOF identik (split fold + init model direproduksi) --
    properti penting utk reproduksibilitas protokol multi-seed proyek ini.

    torch.manual_seed() DISET EKSPLISIT di sini sebelum tiap panggilan --
    compute_oof_predictions() sendiri TIDAK men-seed torch (konsisten dgn
    pola proyek: seed diset SEKALI di awal run_pipeline(), bukan per-fungsi)
    -- dalam produksi fungsi ini cuma dipanggil sekali per proses, jadi ini
    murni kebutuhan harness test memanggilnya 2x dlm 1 proses yg sama."""
    train_df, val_df, user2idx, item2idx, n_items = synthetic_interactions
    config = DeepMFConfig(embedding_dim=8, hidden_layers=(16, 8), epochs=2, batch_size=32)
    val_dataset = InteractionDataset(val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=42)

    torch.manual_seed(7)
    preds_a = compute_oof_predictions(
        train_df, val_dataset, user2idx, item2idx, n_items, config,
        rating_scale=RATING_SCALE, seed=7, n_folds=3,
    )
    torch.manual_seed(7)
    preds_b = compute_oof_predictions(
        train_df, val_dataset, user2idx, item2idx, n_items, config,
        rating_scale=RATING_SCALE, seed=7, n_folds=3,
    )
    np.testing.assert_allclose(preds_a, preds_b)
