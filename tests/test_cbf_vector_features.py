"""
tests/test_cbf_vector_features.py

Regresi utk CBFPredictor.predict_vector_features() /
predict_vector_features_train_loo() (Stage C, port A2-FusionRS Fase 2) --
versi VEKTOR (bukan skalar rating) dari predict()/predict_train_loo() yang
sudah divalidasi, dibutuhkan Attention-Gated Fusion sbg token modalitas CBF.

Properti yang diverifikasi (bukan akurasi model -- itu diukur end-to-end):
1. Bentuk output benar: (N, item_feature_dim + n_clusters), tanpa NaN.
2. predict_vector_features_train_loo() SUNGGUH menerapkan koreksi LOO pada
   komponen vektor item (bukan cuma skalar) -- disadap via monkeypatch
   feature_builder.transform() persis pola test_cbf_train_loo.py.
3. Fallback cold-start: item tak dikenal -> komponen item vektor NOL.
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

from src.baseline.cbf_clustering import CBFConfig, CBFPredictor  # noqa: E402


def _make_train_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["business_categories"] = None
    df["sentiment_score"] = 0.5  # tidak dipakai (include_sentiment default False)
    return df


@pytest.fixture
def synthetic_train_df():
    """Sama persis fixture tests/test_cbf_train_loo.py -- 20 item, 5
    review/item, 2 topik jelas berbeda supaya clustering (k_min=2) punya
    struktur nyata."""
    rng = np.random.RandomState(42)
    rows = []
    topic_a_words = ["battery", "screen", "durable", "charger", "laptop"]
    topic_b_words = ["delicious", "spicy", "waiter", "restaurant", "menu"]

    for item_idx in range(20):
        is_topic_a = item_idx % 2 == 0
        topic_words = topic_a_words if is_topic_a else topic_b_words
        base_rating = 5.0 if is_topic_a else 1.5
        iid = f"item_{item_idx}"
        n_reviews = 5
        for r in range(n_reviews):
            uid = f"user_{item_idx}_{r}"
            text = " ".join(rng.choice(topic_words, size=6))
            rows.append({
                "review_id": f"{iid}_r{r}",
                "user_id": uid,
                "business_id": iid,
                "text_tfidf": text,
                "stars": base_rating,
            })
    return _make_train_df(rows)


def test_predict_vector_features_shape_and_no_nan(synthetic_train_df):
    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(synthetic_train_df, synthetic_train_df)

    vecs = predictor.predict_vector_features(synthetic_train_df)

    n_clusters = predictor.clusterer.best_k
    assert vecs.shape == (len(synthetic_train_df), predictor._item_feature_dim + n_clusters)
    assert np.isfinite(vecs).all()


def test_predict_vector_features_train_loo_shape_and_no_nan(synthetic_train_df):
    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(synthetic_train_df, synthetic_train_df)

    vecs = predictor.predict_vector_features_train_loo(synthetic_train_df)

    n_clusters = predictor.clusterer.best_k
    assert vecs.shape == (len(synthetic_train_df), predictor._item_feature_dim + n_clusters)
    assert np.isfinite(vecs).all()


def test_cold_start_item_gives_zero_item_component(synthetic_train_df):
    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(synthetic_train_df, synthetic_train_df)

    unseen_df = pd.DataFrame([{
        "review_id": "unseen_r0",
        "user_id": "user_0_0",  # user dikenal, item TIDAK
        "business_id": "never_seen_item",
        "text_tfidf": "irrelevant",
        "stars": 3.0,
        "business_categories": None,
        "sentiment_score": 0.5,
    }])

    vecs = predictor.predict_vector_features(unseen_df)
    item_component = vecs[0, : predictor._item_feature_dim]
    assert np.allclose(item_component, 0.0)


def test_train_loo_vector_excludes_own_row(synthetic_train_df, monkeypatch):
    """Sama pola test_cbf_train_loo.py::test_loo_excludes_own_row_from_item_aggregate
    -- verifikasi arithmetic LOO langsung via spy pada transform(), bukan
    bergantung hasil clustering yang bisa kebetulan."""
    item_a_rating, item_b_rating = 5.0, 1.5
    small_item_rows = pd.DataFrame([
        {
            "review_id": "small_item_a", "user_id": "user_small_a", "business_id": "small_item",
            "text_tfidf": "battery screen durable charger laptop", "stars": item_a_rating,
            "business_categories": None, "sentiment_score": 0.5,
        },
        {
            "review_id": "small_item_b", "user_id": "user_small_b", "business_id": "small_item",
            "text_tfidf": "delicious spicy waiter restaurant menu", "stars": item_b_rating,
            "business_categories": None, "sentiment_score": 0.5,
        },
    ])
    train_df = pd.concat([synthetic_train_df, small_item_rows], ignore_index=True)

    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(train_df, train_df)

    captured_calls = []
    original_transform = predictor.feature_builder.transform

    def spy_transform(item_df):
        captured_calls.append(item_df.copy())
        return original_transform(item_df)

    monkeypatch.setattr(predictor.feature_builder, "transform", spy_transform)
    predictor.predict_vector_features_train_loo(train_df)

    small_item_calls = [df for df in captured_calls if (df["business_id"] == "small_item").all()]
    assert len(small_item_calls) == 1
    mini_df = small_item_calls[0].reset_index(drop=True)
    assert mini_df.loc[0, "avg_rating"] == pytest.approx(item_b_rating)
    assert mini_df.loc[1, "avg_rating"] == pytest.approx(item_a_rating)
    assert "battery" not in mini_df.loc[0, "description_text"]
    assert "delicious" not in mini_df.loc[1, "description_text"]


def test_user_pref_component_identical_between_loo_and_plain(synthetic_train_df):
    """Komponen user-preference (bagian kedua vektor) TIDAK bergantung
    koreksi LOO -- lihat docstring _user_pref_vectors(). Verifikasi
    langsung: kolom user-pref plain vs train_loo identik utk baris yang
    sama (item component boleh beda, user component TIDAK)."""
    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(synthetic_train_df, synthetic_train_df)

    plain = predictor.predict_vector_features(synthetic_train_df)
    loo = predictor.predict_vector_features_train_loo(synthetic_train_df)

    d = predictor._item_feature_dim
    np.testing.assert_allclose(plain[:, d:], loo[:, d:])
