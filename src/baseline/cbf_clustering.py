"""
src/baseline/cbf_clustering.py

Reimplementasi Content-Based Filtering dengan clustering, mengikuti
metodologi baseline (K-Means untuk restoran, Agglomerative untuk hotel)
sekaligus proposal (fitur: kategori one-hot, TF-IDF deskripsi, agregasi
sentimen per item, metrik popularitas).

Catatan penting: fitur "agregasi sentimen per item" di sini HARUS berasal
dari output modul sentiment_bert.py pada data TRAIN saja (tidak boleh
memasukkan skor sentimen dari item yang hanya muncul di test set) --
kalau tidak, ini jadi salah satu kemungkinan sumber leakage yang membuat
RMSE baseline paper tampak sangat rendah (lih. diskusi sebelumnya).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class CBFConfig:
    method: str = "kmeans"  # "kmeans" (restoran) atau "agglomerative" (hotel)
    k_min: int = 2
    k_max: int = 20
    tfidf_max_features: int = 500
    # Jumlah komponen PCA sebelum clustering, sesuai pipeline paper (Fig. 5:
    # Concatenation -> Dimensionality reduction (PCA) -> Clustering). Tanpa
    # ini, KMeans/silhouette terdegradasi di ruang fitur berdimensi tinggi
    # (curse of dimensionality) -- dibatasi otomatis ke min(pca_components,
    # n_item-1, n_fitur) saat runtime supaya aman untuk subset kecil.
    pca_components: int = 50
    random_state: int = 42


class ItemFeatureBuilder:
    """Bangun fitur item gabungan: kategori one-hot + TF-IDF + sentimen + popularitas,
    lalu reduksi dimensi dengan PCA sebelum clustering (sesuai Fig. 5 paper)."""

    def __init__(self, tfidf_max_features: int = 500, pca_components: int = 50, random_state: int = 42):
        self.tfidf_max_features = tfidf_max_features
        self.pca_components = pca_components
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self._fitted = False

    def _combine_raw_features(self, cat_features, tfidf_features, numeric_features) -> np.ndarray:
        return np.concatenate([cat_features, tfidf_features, numeric_features], axis=1)

    def fit_transform(self, item_df: pd.DataFrame) -> np.ndarray:
        """
        item_df harus punya kolom:
        - business_id
        - categories_list: list[str] hasil split business_categories
        - description_text: teks deskripsi/gabungan review untuk TF-IDF
        - sentiment_agg: rata-rata skor sentimen dari REVIEW TRAIN SAJA
        - review_count, avg_rating: metrik popularitas
        """
        cat_features = self.mlb.fit_transform(item_df["categories_list"])
        if cat_features.shape[1] == 0:
            logger.info(
                "categories_list kosong utk semua item (domain tanpa metadata kategori, "
                "mis. Amazon/TripAdvisor) -- fitur kategori 0-dim, CBF mundur ke TF-IDF+numerik saja."
            )
        tfidf_features = self.tfidf.fit_transform(item_df["description_text"]).toarray()

        numeric_cols = ["sentiment_agg", "review_count", "avg_rating"]
        numeric_features = self.scaler.fit_transform(item_df[numeric_cols].values)

        combined = self._combine_raw_features(cat_features, tfidf_features, numeric_features)

        # PCA sebelum clustering (Fig. 5 paper: Concatenation -> Dimensionality
        # reduction -> Clustering). n_components dibatasi otomatis supaya aman
        # untuk subset kecil (mis. quicktest dengan sedikit item).
        n_components = max(1, min(self.pca_components, combined.shape[0] - 1, combined.shape[1]))
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = self.pca.fit_transform(combined)

        self._fitted = True
        logger.info(
            "Fitur item dibangun: %d kategori + %d TF-IDF + %d numerik = dim %d -> "
            "PCA %d dim (explained variance ratio=%.3f)",
            cat_features.shape[1],
            tfidf_features.shape[1],
            numeric_features.shape[1],
            combined.shape[1],
            reduced.shape[1],
            self.pca.explained_variance_ratio_.sum(),
        )
        return reduced

    def transform(self, item_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Panggil fit_transform() pada data train dahulu.")
        cat_features = self.mlb.transform(item_df["categories_list"])
        tfidf_features = self.tfidf.transform(item_df["description_text"]).toarray()
        numeric_cols = ["sentiment_agg", "review_count", "avg_rating"]
        numeric_features = self.scaler.transform(item_df[numeric_cols].values)
        combined = self._combine_raw_features(cat_features, tfidf_features, numeric_features)
        return self.pca.transform(combined)


class ContentBasedClusterer:
    def __init__(self, config: CBFConfig | None = None):
        self.config = config or CBFConfig()
        self.model = None
        self.best_k: int | None = None

    def fit(self, item_features: np.ndarray) -> np.ndarray:
        """Cari K optimal via elbow/silhouette, lalu fit model final.

        Untuk Agglomerative, "elbow" klasik (inertia) tidak tersedia karena
        tidak ada konsep centroid/inertia -- di sini dipakai silhouette
        score sebagai kriteria seleksi K untuk kedua metode agar konsisten.
        """
        best_score = -1.0
        best_k = self.config.k_min
        best_labels = None

        for k in range(self.config.k_min, self.config.k_max + 1):
            if self.config.method == "kmeans":
                model = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            elif self.config.method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=k)
            else:
                raise ValueError(f"method '{self.config.method}' tidak dikenal")

            labels = model.fit_predict(item_features)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(item_features, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                self.model = model

        self.best_k = best_k
        logger.info(
            "K optimal terpilih: %d (silhouette=%.4f) via metode '%s'",
            best_k,
            best_score,
            self.config.method,
        )
        return best_labels

    def predict_user_cluster_preference(
        self, interactions: pd.DataFrame, item_cluster_labels: dict
    ) -> pd.DataFrame:
        """Hitung distribusi preferensi user terhadap cluster (Persamaan 2 proposal):
        P(cluster|user) = sum(rating user pada item di cluster) / sum(semua rating user)
        """
        df = interactions.copy()
        df["cluster"] = df["business_id"].map(item_cluster_labels)

        user_cluster_sum = df.groupby(["user_id", "cluster"])["stars"].sum().reset_index()
        user_total = df.groupby("user_id")["stars"].sum().rename("total_rating")

        user_cluster_pref = user_cluster_sum.merge(user_total, on="user_id")
        user_cluster_pref["preference"] = (
            user_cluster_pref["stars"] / user_cluster_pref["total_rating"]
        )
        return user_cluster_pref[["user_id", "cluster", "preference"]]


def build_item_dataframe(
    full_df: pd.DataFrame, train_df: pd.DataFrame, sentiment_col: str = "sentiment_score"
) -> pd.DataFrame:
    """Bangun item_df untuk ItemFeatureBuilder, dengan kontrol anti-leakage:
    - categories_list: berasal dari `full_df` (atribut bisnis statis, business_categories
      tidak berubah antar baris review yang sama, jadi aman dipakai dari mana saja).
    - description_text: HANYA dari teks review TRAIN (mencegah TF-IDF "melihat"
      kata-kata dari review test).
    - sentiment_agg: HANYA rata-rata sentiment_score dari review TRAIN.
    - review_count, avg_rating: dihitung dari TRAIN saja (bukan dari kolom
      business_review_count/business_stars bawaan dataset, karena kolom itu
      berpotensi sudah mengagregasi review test juga -- lihat catatan di
      docstring modul ini).

    Item yang HANYA muncul di test set (tidak ada di train sama sekali) akan
    tetap masuk daftar (agar clustering/lookup tidak KeyError), tapi dengan
    description_text kosong dan sentiment_agg/review_count/avg_rating diisi
    nilai rata-rata global train sebagai fallback cold-start.
    """
    if "categories_list" not in full_df.columns:
        full_df = full_df.copy()
        full_df["categories_list"] = full_df["business_categories"].fillna("").apply(
            lambda s: [c.strip() for c in s.split(";") if c.strip()]
        )

    all_items = full_df["business_id"].unique()
    item_categories = (
        full_df.drop_duplicates("business_id").set_index("business_id")["categories_list"]
    )

    train_agg = train_df.groupby("business_id").agg(
        description_text=("text_tfidf", lambda x: " ".join(x)),
        sentiment_agg=(sentiment_col, "mean"),
        review_count=("stars", "count"),
        avg_rating=("stars", "mean"),
    )

    global_sentiment_mean = train_df[sentiment_col].mean()
    global_avg_rating = train_df["stars"].mean()

    item_df = pd.DataFrame({"business_id": all_items})
    item_df["categories_list"] = item_df["business_id"].map(item_categories)
    item_df = item_df.merge(train_agg, on="business_id", how="left")

    n_cold_start_items = item_df["description_text"].isna().sum()
    if n_cold_start_items > 0:
        logger.warning(
            "%d item hanya muncul di luar train set (cold-start item) -- "
            "diisi fallback (description kosong, sentiment/rating rata-rata global train).",
            n_cold_start_items,
        )

    item_df["description_text"] = item_df["description_text"].fillna("")
    item_df["sentiment_agg"] = item_df["sentiment_agg"].fillna(global_sentiment_mean)
    item_df["review_count"] = item_df["review_count"].fillna(0)
    item_df["avg_rating"] = item_df["avg_rating"].fillna(global_avg_rating)

    return item_df


class CBFPredictor:
    """Wrapper end-to-end: bangun fitur item -> clustering -> user preference
    -> prediksi rating (Persamaan 2 proposal, disederhanakan -- lihat catatan
    di predict()).

    ASUMSI DESAIN (perlu divalidasi ulang terhadap detail lengkap baseline
    paper): karena item direpresentasikan sebagai satu cluster (hard
    assignment, bukan soft/fuzzy), cosine similarity antara vektor preferensi
    user (distribusi atas semua cluster) dengan vektor one-hot cluster item
    secara matematis tereduksi menjadi nilai preferensi user pada cluster
    tersebut. Implementasi ini memakai reduksi tsb secara eksplisit,
    didokumentasikan sebagai simplifikasi -- BUKAN diklaim identik dengan
    detail eksak baseline paper.
    """

    def __init__(self, cbf_config: CBFConfig | None = None, tfidf_max_features: int = 500):
        self.cbf_config = cbf_config or CBFConfig()
        self.feature_builder = ItemFeatureBuilder(
            tfidf_max_features=tfidf_max_features,
            pca_components=self.cbf_config.pca_components,
            random_state=self.cbf_config.random_state,
        )
        self.clusterer = ContentBasedClusterer(self.cbf_config)
        self.item_cluster_labels: dict | None = None
        self.user_cluster_pref: pd.DataFrame | None = None
        self.cluster_avg_rating: dict | None = None
        self.global_mean_rating: float | None = None

    def fit(self, full_df: pd.DataFrame, train_df: pd.DataFrame) -> None:
        item_df = build_item_dataframe(full_df, train_df)
        item_features = self.feature_builder.fit_transform(item_df)

        labels = self.clusterer.fit(item_features)
        self.item_cluster_labels = dict(zip(item_df["business_id"], labels))

        self.user_cluster_pref = self.clusterer.predict_user_cluster_preference(
            train_df, self.item_cluster_labels
        )

        train_df_c = train_df.copy()
        train_df_c["cluster"] = train_df_c["business_id"].map(self.item_cluster_labels)
        self.cluster_avg_rating = train_df_c.groupby("cluster")["stars"].mean().to_dict()
        self.global_mean_rating = float(train_df["stars"].mean())

    def predict(self, df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)) -> np.ndarray:
        if self.item_cluster_labels is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict().")

        rating_min, rating_max = rating_scale
        pref_lookup = self.user_cluster_pref.set_index(["user_id", "cluster"])["preference"].to_dict()

        preds = np.empty(len(df), dtype=np.float32)
        n_fallback = 0

        for idx, row in enumerate(df.itertuples(index=False)):
            item_id = getattr(row, "business_id")
            user_id = getattr(row, "user_id")
            cluster = self.item_cluster_labels.get(item_id)

            if cluster is None:
                preds[idx] = self.global_mean_rating
                n_fallback += 1
                continue

            preference = pref_lookup.get((user_id, cluster))
            cluster_avg = self.cluster_avg_rating.get(cluster, self.global_mean_rating)

            if preference is None:
                # user belum pernah berinteraksi dengan cluster ini di train
                # (termasuk cold-start user) -> fallback ke rata-rata cluster
                preds[idx] = cluster_avg
                n_fallback += 1
            else:
                # blend preferensi user (dinormalisasi ke skala rating) dengan
                # rata-rata cluster, agar prediksi tetap berada di rentang
                # rating yang masuk akal alih-alih preference mentah (0-1)
                preds[idx] = rating_min + preference * (rating_max - rating_min) * 0.5 + cluster_avg * 0.5

        if n_fallback > 0:
            logger.info(
                "%d/%d baris memakai fallback (user/item baru di cluster) saat prediksi CBF",
                n_fallback,
                len(df),
            )
        return np.clip(preds, rating_min, rating_max)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton CBF clustering -- pastikan sentiment_agg dihitung HANYA dari "
        "data train sebelum dipakai membangun fitur item (cegah leakage)."
    )
