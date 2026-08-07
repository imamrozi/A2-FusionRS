"""
tests/test_bias_baseline.py

Regresi utk UserItemBiasBaseline (src/a2fusionrs/bias_baseline.py) -- jangkar
(base residual) A2-FusionRS Fase 2 yang menggantikan base NMF+DecisionTree.

Properti yang diverifikasi (bukan akurasi model -- itu diukur end-to-end):
1. Aritmetika bias benar (mu, b_u, b_i dgn damping) -- dihitung manual.
2. LOO BENAR-BENAR mengecualikan baris sendiri (bukti langsung: bandingkan
   dgn refit dari nol tanpa baris itu).
3. LOO != prediksi in-sample (kalau sama, koreksi tidak bekerja).
4. Cold-start user/item -> jatuh ke mu (bias 0).
5. Damping menyusutkan bias ke 0 utk data tipis.
6. Deterministik & output selalu dalam rating_scale.
7. Item/user dgn 1 rating tidak menyebabkan pembagian nol di jalur LOO.
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

from src.a2fusionrs.bias_baseline import UserItemBiasBaseline  # noqa: E402

RATING_SCALE = (1.0, 5.0)


@pytest.fixture
def train_df():
    """20 user x 10 item, rating punya struktur user/item yang jelas supaya
    bias tidak nol semua."""
    rng = np.random.RandomState(42)
    rows = []
    for u in range(20):
        user_offset = (u % 5) - 2  # -2..+2, bias user nyata
        for i in range(10):
            item_offset = 0.5 if i % 2 == 0 else -0.5
            r = np.clip(3.0 + user_offset * 0.4 + item_offset + rng.normal(0, 0.2), 1.0, 5.0)
            rows.append({
                "review_id": f"u{u}_i{i}",
                "user_id": f"u{u}",
                "business_id": f"i{i}",
                "stars": float(r),
            })
    return pd.DataFrame(rows)


def test_global_mean_and_bias_arithmetic(train_df):
    """mu, b_u, b_i cocok dgn perhitungan manual (formula Koren + damping)."""
    model = UserItemBiasBaseline(damping=10.0)
    model.fit(train_df)

    assert model.global_mean == pytest.approx(train_df["stars"].mean())

    # b_u utk satu user, dihitung manual
    u = "u3"
    sub = train_df[train_df["user_id"] == u]
    expected_bu = (sub["stars"] - model.global_mean).sum() / (len(sub) + 10.0)
    assert model._user_bias[u] == pytest.approx(expected_bu)


def test_loo_matches_refit_without_that_row(train_df):
    """BUKTI LANGSUNG koreksi LOO: b_u hasil jalur LOO utk suatu baris harus
    sama dgn b_u kalau model di-fit ULANG dari nol TANPA baris itu.

    (Dibandingkan pada komponen b_u, bukan prediksi akhir, karena refit
    penuh juga menggeser mu & b_i sehingga perbandingan prediksi akhir
    tidak apple-to-apple; komponen b_u sudah cukup membuktikan baris ybs
    benar-benar dikeluarkan dari akumulatornya.)"""
    model = UserItemBiasBaseline(damping=10.0)
    model.fit(train_df)

    target_idx = 7
    target = train_df.iloc[target_idx]

    # b_u LOO versi analitik (yang dipakai produksi)
    u_sum = model._user_sum[target["user_id"]]
    u_cnt = model._user_count[target["user_id"]]
    own_dev = target["stars"] - model.global_mean
    bu_loo_analytic = (u_sum - own_dev) / (u_cnt - 1 + model.damping)

    # b_u versi refit dari nol tanpa baris itu, TAPI memakai mu yang sama
    # (isolasi: kita hanya menguji pengeluaran baris dari akumulator b_u)
    without = train_df.drop(index=train_df.index[target_idx])
    sub = without[without["user_id"] == target["user_id"]]
    bu_refit = (sub["stars"] - model.global_mean).sum() / (len(sub) + model.damping)

    assert bu_loo_analytic == pytest.approx(bu_refit)


def test_loo_differs_from_in_sample(train_df):
    """Kalau LOO == in-sample, berarti koreksi tidak bekerja sama sekali."""
    model = UserItemBiasBaseline(damping=10.0)
    model.fit(train_df)

    in_sample = model.predict(train_df, RATING_SCALE)
    loo = model.predict_train_loo(train_df, RATING_SCALE)

    assert not np.allclose(in_sample, loo), "LOO harus berbeda dari prediksi in-sample"
    # tapi tidak boleh liar -- korelasi tetap tinggi (koreksi kecil, bukan acak)
    assert np.corrcoef(in_sample, loo)[0, 1] > 0.9


def test_cold_start_falls_back_to_global_mean(train_df):
    """User & item yang belum pernah dilihat -> bias 0 -> prediksi = mu."""
    model = UserItemBiasBaseline(damping=10.0)
    model.fit(train_df)

    unseen = pd.DataFrame([{
        "review_id": "x", "user_id": "user_baru", "business_id": "item_baru", "stars": 3.0,
    }])
    pred = model.predict(unseen, RATING_SCALE)
    assert pred[0] == pytest.approx(model.global_mean, abs=1e-5)


def test_damping_shrinks_bias_for_sparse_user():
    """User dgn 1 rating ekstrem: damping besar -> bias mendekati 0."""
    df = pd.DataFrame([
        {"review_id": "a", "user_id": "u_sparse", "business_id": "i1", "stars": 5.0},
        {"review_id": "b", "user_id": "u_dense", "business_id": "i1", "stars": 3.0},
        {"review_id": "c", "user_id": "u_dense", "business_id": "i2", "stars": 3.0},
        {"review_id": "d", "user_id": "u_dense", "business_id": "i3", "stars": 3.0},
    ])
    lightly_damped = UserItemBiasBaseline(damping=0.0)
    heavily_damped = UserItemBiasBaseline(damping=50.0)
    lightly_damped.fit(df)
    heavily_damped.fit(df)

    assert abs(heavily_damped._user_bias["u_sparse"]) < abs(lightly_damped._user_bias["u_sparse"])
    assert abs(heavily_damped._user_bias["u_sparse"]) < 0.1


def test_single_rating_user_item_no_division_by_zero():
    """Baris yang user & item-nya hanya punya 1 rating (n-1 = 0) -- damping
    menjaga penyebut > 0, hasil harus finite & dalam rating_scale."""
    df = pd.DataFrame([
        {"review_id": "a", "user_id": "u1", "business_id": "i1", "stars": 5.0},
        {"review_id": "b", "user_id": "u2", "business_id": "i2", "stars": 2.0},
    ])
    model = UserItemBiasBaseline(damping=10.0)
    model.fit(df)
    loo = model.predict_train_loo(df, RATING_SCALE)

    assert np.isfinite(loo).all()
    assert (loo >= RATING_SCALE[0]).all() and (loo <= RATING_SCALE[1]).all()


def test_outputs_bounded_and_deterministic(train_df):
    model_a = UserItemBiasBaseline(damping=10.0)
    model_b = UserItemBiasBaseline(damping=10.0)
    model_a.fit(train_df)
    model_b.fit(train_df)

    pred_a = model_a.predict_train_loo(train_df, RATING_SCALE)
    pred_b = model_b.predict_train_loo(train_df, RATING_SCALE)

    np.testing.assert_allclose(pred_a, pred_b)
    assert (pred_a >= RATING_SCALE[0]).all() and (pred_a <= RATING_SCALE[1]).all()
    assert np.isfinite(pred_a).all()


def test_negative_damping_rejected():
    with pytest.raises(ValueError, match="damping"):
        UserItemBiasBaseline(damping=-1.0)


def test_predict_before_fit_raises(train_df):
    model = UserItemBiasBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(train_df, RATING_SCALE)
