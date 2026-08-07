"""
tests/test_split_train_fit_dev.py

Regresi utk split_train_fit_dev() -- Fix Temuan A2 audit metodologi
(reports/methodology_audit_2026-07-26.md): pisah train_df jadi train_fit
(fitting) + selection_dev (seleksi kandidat, independen dari val
early-stopping).

CATATAN: fungsi ini DIPINDAH dari scripts/tune_deepmf_oof_val.py ke
src/a2fusionrs/selection_split.py (2026-08-07) supaya bisa dipakai ulang
oleh run_attention_gated_fusion.py utk seleksi arsitektur A2-FusionRS
Fase 2. Import di sini disederhanakan jadi import modul biasa (tidak lagi
perlu importlib hack utk memuat script level-atas); PERILAKU fungsi tidak
berubah, jadi seluruh assert di bawah tetap sama persis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.selection_split import split_train_fit_dev  # noqa: E402


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "review_id": [f"r{i}" for i in range(1000)],
        "user_id": [f"u{i % 50}" for i in range(1000)],
        "business_id": [f"b{i % 30}" for i in range(1000)],
        "stars": [float((i % 5) + 1) for i in range(1000)],
    })


def test_sizes_match_fraction(train_df):
    train_fit, selection_dev = split_train_fit_dev(train_df, seed=42, dev_fraction=0.15)
    assert len(selection_dev) == 150
    assert len(train_fit) == 850
    assert len(train_fit) + len(selection_dev) == len(train_df)


def test_no_row_overlap(train_df):
    train_fit, selection_dev = split_train_fit_dev(train_df, seed=42, dev_fraction=0.15)
    fit_ids = set(train_fit["review_id"])
    dev_ids = set(selection_dev["review_id"])
    assert fit_ids.isdisjoint(dev_ids)
    assert fit_ids | dev_ids == set(train_df["review_id"])


def test_deterministic_same_seed(train_df):
    fit_a, dev_a = split_train_fit_dev(train_df, seed=7, dev_fraction=0.15)
    fit_b, dev_b = split_train_fit_dev(train_df, seed=7, dev_fraction=0.15)
    assert list(fit_a["review_id"]) == list(fit_b["review_id"])
    assert list(dev_a["review_id"]) == list(dev_b["review_id"])


def test_different_seed_gives_different_split(train_df):
    _, dev_a = split_train_fit_dev(train_df, seed=1, dev_fraction=0.15)
    _, dev_b = split_train_fit_dev(train_df, seed=2, dev_fraction=0.15)
    assert set(dev_a["review_id"]) != set(dev_b["review_id"])


def test_original_columns_preserved(train_df):
    train_fit, selection_dev = split_train_fit_dev(train_df, seed=42, dev_fraction=0.15)
    assert list(train_fit.columns) == list(train_df.columns)
    assert list(selection_dev.columns) == list(train_df.columns)
