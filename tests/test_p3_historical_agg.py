"""
tests/test_p3_historical_agg.py

Regresi utk compute_p3_features() (src/a2fusionrs/absa_bert.py) -- arm P3
(docs/phase1_spec.md di branch phase2-a2-fusionrs, Step 4): ganti fitur
sentimen/ABSA per-baris dgn profil rata-rata item dari review TRAIN (LOO
utk baris train).

Fungsi ini murni aritmetika pandas/numpy (tidak ada randomness/clustering),
jadi -- BEDA dgn test_cbf_train_loo.py yang harus menghindari perbandingan
end-to-end krn clustering rapuh di data sintetik kecil -- di sini assert
NILAI EKSAK langsung terhadap fungsi di bawah test aman dan tidak flaky.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.absa_bert import compute_p3_features  # noqa: E402


@pytest.fixture
def train_df():
    # item A: 3 review (skor 0.2, 0.4, 0.6) -> mean train = 0.4
    # item B: 2 review (skor 1.0, 0.0) -> mean train = 0.5
    # item C: 1 review (skor 0.9) -- kasus fallback LOO (cuma 1 review)
    return pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
        "business_id": ["A", "A", "A", "B", "B", "C"],
        "score": [0.2, 0.4, 0.6, 1.0, 0.0, 0.9],
    })


def test_eval_no_loo_uses_full_train_mean(train_df):
    """Baris eval (test set): profil = rata-rata SEMUA review train item
    itu, tanpa exclude apa pun (test review tidak pernah ada di train)."""
    eval_df = pd.DataFrame({"business_id": ["A", "B"]})
    profile = compute_p3_features(train_df, eval_df, ["score"], exclude_own_row=False)

    np.testing.assert_allclose(profile[:, 0], [0.4, 0.5])


def test_train_loo_excludes_own_row(train_df):
    """Baris train: profil HARUS mengecualikan review baris itu sendiri --
    utk item A baris pertama (skor 0.2), profil = mean(0.4, 0.6) = 0.5."""
    profile = compute_p3_features(train_df, train_df, ["score"], exclude_own_row=True)

    # item A: r1(0.2)->mean(0.4,0.6)=0.5 ; r2(0.4)->mean(0.2,0.6)=0.4 ; r3(0.6)->mean(0.2,0.4)=0.3
    # item B: r4(1.0)->mean(0.0)=0.0 ; r5(0.0)->mean(1.0)=1.0
    expected_non_fallback = {
        0: 0.5,  # r1
        1: 0.4,  # r2
        2: 0.3,  # r3
        3: 0.0,  # r4
        4: 1.0,  # r5
    }
    for row_idx, expected in expected_non_fallback.items():
        assert profile[row_idx, 0] == pytest.approx(expected)


def test_train_loo_single_review_item_falls_back_to_global_mean(train_df):
    """Item C cuma py 1 review train -> setelah LOO, tidak ada review
    'lain' sbg profil -> fallback ke rata-rata GLOBAL train (bukan NaN)."""
    profile = compute_p3_features(train_df, train_df, ["score"], exclude_own_row=True)

    global_mean = train_df["score"].mean()
    assert profile[5, 0] == pytest.approx(global_mean)  # r6, item C


def test_eval_cold_start_item_falls_back_to_global_mean(train_df):
    """Item eval yang TIDAK pernah muncul di train sama sekali (cold-start)
    -> fallback ke rata-rata global train, bukan NaN/error."""
    eval_df = pd.DataFrame({"business_id": ["Z"]})
    profile = compute_p3_features(train_df, eval_df, ["score"], exclude_own_row=False)

    global_mean = train_df["score"].mean()
    assert profile[0, 0] == pytest.approx(global_mean)


def test_multi_column_features_vectorized_correctly():
    """feature_cols > 1 kolom (spt mode concat_confidence: skor + confidence)
    -- tiap kolom diagregasi independen, LOO tidak saling mencampur kolom."""
    train_df = pd.DataFrame({
        "business_id": ["A", "A", "B", "B"],
        "aspect1": [0.2, 0.8, 1.0, 0.0],
        "aspect2": [0.5, 0.5, 0.9, 0.1],
    })
    profile = compute_p3_features(
        train_df, train_df, ["aspect1", "aspect2"], exclude_own_row=True,
    )
    # item A baris 0: LOO -> hanya baris 1 tersisa -> (0.8, 0.5)
    np.testing.assert_allclose(profile[0], [0.8, 0.5])
    # item B baris 2: LOO -> hanya baris 3 tersisa -> (0.0, 0.1)
    np.testing.assert_allclose(profile[2], [0.0, 0.1])


def test_output_shape_matches_eval_df(train_df):
    eval_df = pd.DataFrame({"business_id": ["A", "B", "C", "Z"]})
    profile = compute_p3_features(train_df, eval_df, ["score"], exclude_own_row=False)
    assert profile.shape == (4, 1)
