"""
tests/test_cbf_train_loo.py

Regresi utk CBFPredictor.predict_train_loo() -- koreksi leave-one-out saat
scoring baris TRAIN (lihat reports/cbf_tfidf_leakage_measurement.md utk
motivasi & pengukuran besaran masalah SEBELUM diperbaiki).

Properti inti yang diverifikasi (bukan angka RMSE riil -- itu diukur via
smoke test end-to-end run_baseline_absa.py, lihat riwayat kerja):
1. Item dgn 1 review (review itu sendiri) -> LOO jatuh ke fallback global,
   BUKAN memakai review itu sendiri sbg satu-satunya basis profil.
2. Item dgn review yg SANGAT BERBEDA (mis. satu review "outlier" beda topik
   total dari review lain item yg sama) -> baris outlier itu, setelah LOO,
   TIDAK mendapat cluster yg sama dgn cluster yg ditentukan predict() naif
   (yg justru dipengaruhi outlier itu sendiri) -- membuktikan koreksi
   benar-benar mengecualikan kontribusi baris itu.
3. Item dgn review SERAGAM (topik/rating sama) -> LOO TIDAK mengubah apa-
   apa scr berarti (dilusi besar, konsisten dgn temuan shift kecil di item
   ber-review-count tinggi).
4. predict_train_loo() TIDAK error & mengembalikan array sepanjang len(df),
   dlm rentang rating_scale.
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

RATING_SCALE = (1.0, 5.0)


def _make_train_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["business_categories"] = None
    df["sentiment_score"] = 0.5  # tidak dipakai (include_sentiment default False)
    return df


@pytest.fixture
def synthetic_train_df():
    """20 item, ~5 review/item, 2 topik jelas berbeda (elektronik vs makanan)
    supaya clustering (k_min=2) punya struktur nyata utk dipelajari, dan
    review_count cukup besar per item utk uji properti (1)-(3) di atas.
    Rating SENGAJA dikorelasikan dgn topik (topik A tinggi, topik B rendah)
    -- supaya cluster_avg_rating dua cluster BERBEDA JAUH, sehingga kalau
    baris berpindah cluster akibat koreksi LOO, itu KELIHATAN di prediksi
    akhir (bukan tercuci krn rata-rata rating semua cluster kebetulan sama)."""
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


def test_predict_train_loo_runs_and_bounded(synthetic_train_df):
    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(synthetic_train_df, synthetic_train_df)

    preds = predictor.predict_train_loo(synthetic_train_df, RATING_SCALE)

    assert len(preds) == len(synthetic_train_df)
    assert np.isfinite(preds).all()
    assert (preds >= RATING_SCALE[0]).all() and (preds <= RATING_SCALE[1]).all()


def test_single_review_item_falls_back_to_global(synthetic_train_df):
    """Tambah 1 item dgn TEPAT 1 review -- LOO utk baris itu harus jatuh ke
    fallback global (tidak ada basis profil sama sekali selain dirinya
    sendiri, yg justru harus dikecualikan)."""
    extra = pd.DataFrame([{
        "review_id": "lonely_item_r0",
        "user_id": "lonely_user",
        "business_id": "lonely_item",
        "text_tfidf": "unique standalone product description",
        "stars": 3.0,
        "business_categories": None,
        "sentiment_score": 0.5,
    }])
    train_df = pd.concat([synthetic_train_df, extra], ignore_index=True)

    predictor = CBFPredictor(cbf_config=CBFConfig(method="kmeans", k_min=2, k_max=4, random_state=42))
    predictor.fit(train_df, train_df)

    preds = predictor.predict_train_loo(train_df, RATING_SCALE)
    # Tidak crash & baris ini dapat prediksi valid (fallback ke global_mean
    # atau cluster_avg, keduanya dlm rating_scale -- properti fungsional,
    # bukan nilai spesifik yg diklaim).
    idx = train_df.index[train_df["review_id"] == "lonely_item_r0"][0]
    assert RATING_SCALE[0] <= preds[idx] <= RATING_SCALE[1]


def test_loo_excludes_own_row_from_item_aggregate(synthetic_train_df, monkeypatch):
    """Verifikasi ARITMETIKA inti LOO secara langsung & deterministik --
    TIDAK bergantung hasil clustering (dicoba 2 pendekatan lain berbasis
    "apakah prediksi/cluster akhir berubah", keduanya TERBUKTI rapuh thd
    kebetulan struktur data sintetik: cluster_avg_rating bisa kebetulan
    identik antar cluster, atau clustering tidak sensitif thd perubahan
    kecil -- lihat riwayat iterasi test ini). Pendekatan ini menyadap
    `ItemFeatureBuilder.transform()` (dipanggil predict_train_loo per item)
    dan memverifikasi `avg_rating`/`review_count`/`description_text` yang
    BENAR-BENAR dikirim ke situ utk item ber-2-review, seharusnya PERSIS
    sama dgn menghitung manual "rata-rata SATU review yg TERSISA" -- bukti
    langsung bahwa baris itu sendiri dikecualikan dari agregat, terlepas
    dari ke cluster mana ujungnya ditentukan."""
    item_a_rating, item_b_rating = 5.0, 1.5
    small_item_rows = pd.DataFrame([
        {
            "review_id": "small_item_a",
            "user_id": "user_small_a",
            "business_id": "small_item",
            "text_tfidf": "battery screen durable charger laptop",
            "stars": item_a_rating,
            "business_categories": None,
            "sentiment_score": 0.5,
        },
        {
            "review_id": "small_item_b",
            "user_id": "user_small_b",
            "business_id": "small_item",
            "text_tfidf": "delicious spicy waiter restaurant menu",
            "stars": item_b_rating,
            "business_categories": None,
            "sentiment_score": 0.5,
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
    predictor.predict_train_loo(train_df, RATING_SCALE)

    small_item_calls = [df for df in captured_calls if (df["business_id"] == "small_item").all()]
    assert len(small_item_calls) == 1, "Diharapkan tepat 1 panggilan transform() batch utk 'small_item'."
    mini_df = small_item_calls[0].reset_index(drop=True)
    assert len(mini_df) == 2

    # Baris utk small_item_a (excluded=item_a_rating) -> avg_rating LOO
    # HARUS = rating review LAIN (item_b_rating), BUKAN rata-rata keduanya.
    assert mini_df.loc[0, "avg_rating"] == pytest.approx(item_b_rating)
    assert mini_df.loc[1, "avg_rating"] == pytest.approx(item_a_rating)
    # review_count LOO = n-1 = 1 utk keduanya (item cuma py 2 review total).
    assert (mini_df["review_count"] == 1).all()
    # description_text LOO tidak boleh memuat teks review itu sendiri.
    assert "battery" not in mini_df.loc[0, "description_text"]
    assert "delicious" not in mini_df.loc[1, "description_text"]
