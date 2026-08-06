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
    # Default False = desain yang BENAR sesuai arsitektur A2-IRM: sentiment
    # analysis (SA global maupun ABSA) TERPISAH dari CF dan CBF, langsung
    # masuk ke Fusion. sentiment_agg TIDAK termasuk fitur numerik item CBF
    # (lihat ItemFeatureBuilder). True TERSEDIA hanya utk mereproduksi
    # perilaku LAMA (sebelum diperbaiki 2026-07-25) sbg pembanding/
    # dokumentasi historis -- BUKAN varian yang direkomendasikan. Diverifikasi
    # (reports/absa_aggregation_comparison.md): keluarkan sentiment dari CBF
    # tidak mengorbankan performa berarti di 13/15 kombinasi model x domain.
    include_sentiment: bool = False


class ItemFeatureBuilder:
    """Bangun fitur item gabungan: kategori one-hot + TF-IDF + sentimen + popularitas,
    lalu reduksi dimensi dengan PCA sebelum clustering (sesuai Fig. 5 paper)."""

    def __init__(
        self,
        tfidf_max_features: int = 500,
        pca_components: int = 50,
        random_state: int = 42,
        include_sentiment: bool = False,
    ):
        self.tfidf_max_features = tfidf_max_features
        self.pca_components = pca_components
        self.random_state = random_state
        self.include_sentiment = include_sentiment
        self.mlb = MultiLabelBinarizer()
        self.tfidf = TfidfVectorizer(max_features=tfidf_max_features)
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self._fitted = False

    def _numeric_cols(self) -> list[str]:
        # include_sentiment=False -> ablasi: sentiment_agg dikeluarkan dari
        # fitur numerik item CBF (Invarian #9: kebijakan eksplisit, bukan
        # default tersembunyi -- lihat CBFConfig.include_sentiment).
        cols = ["review_count", "avg_rating"]
        return (["sentiment_agg"] + cols) if self.include_sentiment else cols

    def _combine_raw_features(self, cat_features, tfidf_features, numeric_features) -> np.ndarray:
        return np.concatenate([cat_features, tfidf_features, numeric_features], axis=1)

    def fit_transform(self, item_df: pd.DataFrame) -> np.ndarray:
        """
        item_df harus punya kolom:
        - business_id
        - categories_list: list[str] hasil split business_categories
        - description_text: teks deskripsi/gabungan review untuk TF-IDF
        - sentiment_agg: rata-rata skor sentimen dari REVIEW TRAIN SAJA
          (diabaikan kalau include_sentiment=False)
        - review_count, avg_rating: metrik popularitas
        """
        cat_features = self.mlb.fit_transform(item_df["categories_list"])
        if cat_features.shape[1] == 0:
            logger.info(
                "categories_list kosong utk semua item (domain tanpa metadata kategori, "
                "mis. Amazon/TripAdvisor) -- fitur kategori 0-dim, CBF mundur ke TF-IDF+numerik saja."
            )
        tfidf_features = self.tfidf.fit_transform(item_df["description_text"]).toarray()

        numeric_cols = self._numeric_cols()
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
        numeric_cols = self._numeric_cols()
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
            include_sentiment=self.cbf_config.include_sentiment,
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

        # ---- State utk predict_train_loo() (lihat docstring method itu) ----
        self._item_categories = item_df.set_index("business_id")["categories_list"]
        labels_arr = np.asarray(labels)
        self._cluster_centroids = {
            int(c): item_features[labels_arr == c].mean(axis=0) for c in np.unique(labels_arr)
        }
        # ---- State utk predict_vector_features() (Fase 2 A2-FusionRS,
        # lihat docstring method itu) -- item_features di sini SUDAH keluaran
        # fit_transform() (PCA-reduced), sejajar baris dgn item_df["business_id"]
        self._item_feature_vectors = dict(zip(item_df["business_id"], item_features))
        self._item_feature_dim = item_features.shape[1]
        self._global_avg_rating = float(train_df["stars"].mean())
        self._sentiment_col = "sentiment_score"  # SAMA dgn default build_item_dataframe()
        self._global_sentiment_mean = (
            float(train_df[self._sentiment_col].mean())
            if self.feature_builder.include_sentiment
            else None
        )

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

    def predict_train_loo(
        self, train_df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)
    ) -> np.ndarray:
        """Prediksi CBF KHUSUS baris TRAIN, dengan koreksi leave-one-out.

        Motivasi (reports/cbf_tfidf_leakage_measurement.md): `predict()` biasa
        memakai `item_cluster_labels` yang DITENTUKAN SEKALI di fit() dari
        profil item (description_text/avg_rating/review_count) yang mengagre-
        gasi SELURUH review train item itu -- TERMASUK review baris (u,i) yang
        sedang diberi skor. Untuk baris TRAIN, ini artinya cluster item i bisa
        "melihat" review targetnya sendiri. Method ini mengoreksinya: profil
        item dibangun ULANG per baris, MENGECUALIKAN review baris itu (mean
        rating & TF-IDF description tanpa kontribusi baris ybs), lalu
        ditransform lewat vectorizer/scaler/PCA yang SAMA (dari fit(), TIDAK
        di-refit -- basis reduksi dimensi tetap konsisten dgn fit awal),
        dan cluster ditentukan ulang via NEAREST CENTROID (jarak Euclidean ke
        rata-rata fitur tiap cluster hasil fit() asli).

        PENDEKATAN APPROXIMATE YANG DISENGAJA (Invarian #9, dinyatakan
        eksplisit): re-clustering PENUH per baris tidak feasible secara
        komputasi (setiap baris train perlu clustering ulang atas SEMUA
        item). Nearest-centroid adalah aproksimasi murah yang tetap menutup
        celah leakage inti (review target tidak lagi ikut membentuk profil
        yang menentukan cluster baris itu sendiri), TANPA mengklaim setara
        dengan re-clustering global per baris.

        HANYA relevan utk TRAIN -- baris val/test TIDAK PERNAH exposed (review
        mereka tidak pernah ikut membangun item_df sejak awal, lihat
        `build_item_dataframe()`), jadi `predict()` biasa tetap benar & lebih
        murah utk val/test -- JANGAN panggil method ini utk itu.
        """
        if self.item_cluster_labels is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict_train_loo().")

        rating_min, rating_max = rating_scale
        pref_lookup = self.user_cluster_pref.set_index(["user_id", "cluster"])["preference"].to_dict()
        include_sentiment = self.feature_builder.include_sentiment
        sentiment_col = self._sentiment_col

        centroid_ids = list(self._cluster_centroids.keys())
        centroid_matrix = np.stack([self._cluster_centroids[c] for c in centroid_ids])

        preds = np.empty(len(train_df), dtype=np.float32)
        row_pos = {rid: i for i, rid in enumerate(train_df["review_id"].values)}
        n_reassigned = 0
        n_fallback = 0

        for iid, grp in train_df.groupby("business_id"):
            rids = grp["review_id"].tolist()
            texts = grp["text_tfidf"].tolist()
            ratings = grp["stars"].tolist()
            uids = grp["user_id"].tolist()
            sentiments = grp[sentiment_col].tolist() if include_sentiment else None
            n = len(grp)

            if n == 1:
                # Item HANYA punya review baris ini sendiri -- LOO -> tidak
                # ada basis profil item sama sekali, fallback global (SAMA
                # kebijakan cold-start dgn build_item_dataframe()).
                loo_texts = [""]
                loo_avg_ratings = [self._global_avg_rating]
                loo_review_counts = [0]
                loo_sentiments = [self._global_sentiment_mean] if include_sentiment else None
            else:
                total_rating = sum(ratings)
                total_sentiment = sum(sentiments) if sentiments is not None else None
                loo_texts, loo_avg_ratings, loo_review_counts = [], [], []
                loo_sentiments = [] if sentiments is not None else None
                for j in range(n):
                    loo_texts.append(" ".join(texts[:j] + texts[j + 1 :]))
                    loo_avg_ratings.append((total_rating - ratings[j]) / (n - 1))
                    loo_review_counts.append(n - 1)
                    if sentiments is not None:
                        loo_sentiments.append((total_sentiment - sentiments[j]) / (n - 1))

            cat = self._item_categories.get(iid, [])
            mini_df = pd.DataFrame(
                {
                    "business_id": [iid] * n,
                    "categories_list": [cat] * n,
                    "description_text": loo_texts,
                    "review_count": loo_review_counts,
                    "avg_rating": loo_avg_ratings,
                }
            )
            if include_sentiment:
                mini_df["sentiment_agg"] = loo_sentiments

            corrected_features = self.feature_builder.transform(mini_df)  # (n, D)
            dists = np.linalg.norm(
                corrected_features[:, None, :] - centroid_matrix[None, :, :], axis=2
            )
            nearest = dists.argmin(axis=1)
            corrected_clusters = [centroid_ids[k] for k in nearest]

            naive_cluster = self.item_cluster_labels.get(iid)
            for j in range(n):
                cluster = corrected_clusters[j]
                if cluster != naive_cluster:
                    n_reassigned += 1
                preference = pref_lookup.get((uids[j], cluster))
                cluster_avg = self.cluster_avg_rating.get(cluster, self.global_mean_rating)
                if preference is None:
                    pred = cluster_avg
                    n_fallback += 1
                else:
                    pred = rating_min + preference * (rating_max - rating_min) * 0.5 + cluster_avg * 0.5
                preds[row_pos[rids[j]]] = pred

        logger.info(
            "predict_train_loo: %d/%d baris (%.1f%%) berpindah cluster setelah koreksi LOO; "
            "%d/%d baris fallback (kombinasi user x cluster-terkoreksi belum pernah ada di train).",
            n_reassigned,
            len(train_df),
            100.0 * n_reassigned / len(train_df),
            n_fallback,
            len(train_df),
        )
        return np.clip(preds, rating_min, rating_max)

    def _user_pref_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """Vektor preferensi user ATAS SEMUA cluster (bukan cuma cluster yg
        di-assign ke suatu item) -- dipakai bareng vektor fitur item oleh
        predict_vector_features()/predict_vector_features_train_loo() di
        bawah. TIDAK bergantung cluster mana yang di-assign ke item baris
        ybs (independen dari koreksi LOO), jadi aman dipakai sama persis
        oleh kedua varian (train-LOO maupun val/test biasa)."""
        n_clusters = self.clusterer.best_k
        user_vecs = np.zeros((len(df), n_clusters), dtype=np.float32)
        pref_lookup = self.user_cluster_pref.set_index(["user_id", "cluster"])["preference"].to_dict()
        user_ids = df["user_id"].values
        for idx, user_id in enumerate(user_ids):
            for cluster in range(n_clusters):
                pref = pref_lookup.get((user_id, cluster))
                if pref is not None:
                    user_vecs[idx, cluster] = pref
        return user_vecs

    def predict_vector_features(self, df: pd.DataFrame) -> np.ndarray:
        """Kembalikan representasi VEKTOR (bukan skalar rating) per baris:
        konkatenasi [vektor fitur item PCA-reduced] + [vektor preferensi
        user atas SEMUA cluster] -- dibutuhkan Attention-Gated Fusion
        (Fase 2, A2-FusionRS) sebagai salah satu token input attention,
        menggantikan skalar rating akhir yang dipakai fusi statis
        NMF+DecisionTree (Fase 1).

        HANYA utk baris val/test (genuinely out-of-sample, item profile
        TIDAK exposed thd review baris ybs -- lihat build_item_dataframe()).
        Utk baris TRAIN, pakai predict_vector_features_train_loo() di bawah
        (hindari leakage yang sama seperti predict() vs predict_train_loo()
        skalar).

        Fallback cold-start: vektor item NOL kalau item tidak dikenal sama
        sekali; entri preferensi user NOL utk cluster yg belum pernah
        diinteraksi user di train (termasuk user baru -- vektor NOL semua).
        """
        if self.item_cluster_labels is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict_vector_features().")

        item_vecs = np.zeros((len(df), self._item_feature_dim), dtype=np.float32)
        for idx, item_id in enumerate(df["business_id"].values):
            vec = self._item_feature_vectors.get(item_id)
            if vec is not None:
                item_vecs[idx] = vec

        user_vecs = self._user_pref_vectors(df)
        return np.concatenate([item_vecs, user_vecs], axis=1)

    def predict_vector_features_train_loo(self, train_df: pd.DataFrame) -> np.ndarray:
        """Sama dgn predict_vector_features(), TAPI KHUSUS baris TRAIN dgn
        koreksi leave-one-out pada vektor fitur item -- profil item
        dibangun ULANG per baris, MENGECUALIKAN review baris itu sendiri,
        persis mekanisme predict_train_loo() (lihat docstring method itu
        utk motivasi lengkap & pendekatan nearest-centroid approximate yg
        disengaja), TAPI mengembalikan vektor fitur ter-transform (SEBELUM
        cluster hard-assignment), bukan skalar prediksi rating.

        Vektor preferensi user (komponen kedua) TIDAK butuh koreksi LOO --
        independen dari item baris ybs, lihat _user_pref_vectors().
        """
        if self.item_cluster_labels is None:
            raise RuntimeError(
                "Panggil fit() terlebih dahulu sebelum predict_vector_features_train_loo()."
            )

        include_sentiment = self.feature_builder.include_sentiment
        sentiment_col = self._sentiment_col

        item_vecs = np.zeros((len(train_df), self._item_feature_dim), dtype=np.float32)
        row_pos = {rid: i for i, rid in enumerate(train_df["review_id"].values)}

        for iid, grp in train_df.groupby("business_id"):
            rids = grp["review_id"].tolist()
            texts = grp["text_tfidf"].tolist()
            ratings = grp["stars"].tolist()
            sentiments = grp[sentiment_col].tolist() if include_sentiment else None
            n = len(grp)

            if n == 1:
                loo_texts = [""]
                loo_avg_ratings = [self._global_avg_rating]
                loo_review_counts = [0]
                loo_sentiments = [self._global_sentiment_mean] if include_sentiment else None
            else:
                total_rating = sum(ratings)
                total_sentiment = sum(sentiments) if sentiments is not None else None
                loo_texts, loo_avg_ratings, loo_review_counts = [], [], []
                loo_sentiments = [] if sentiments is not None else None
                for j in range(n):
                    loo_texts.append(" ".join(texts[:j] + texts[j + 1 :]))
                    loo_avg_ratings.append((total_rating - ratings[j]) / (n - 1))
                    loo_review_counts.append(n - 1)
                    if sentiments is not None:
                        loo_sentiments.append((total_sentiment - sentiments[j]) / (n - 1))

            cat = self._item_categories.get(iid, [])
            mini_df = pd.DataFrame(
                {
                    "business_id": [iid] * n,
                    "categories_list": [cat] * n,
                    "description_text": loo_texts,
                    "review_count": loo_review_counts,
                    "avg_rating": loo_avg_ratings,
                }
            )
            if include_sentiment:
                mini_df["sentiment_agg"] = loo_sentiments

            corrected_features = self.feature_builder.transform(mini_df)  # (n, D)
            for j in range(n):
                item_vecs[row_pos[rids[j]]] = corrected_features[j]

        user_vecs = self._user_pref_vectors(train_df)
        return np.concatenate([item_vecs, user_vecs], axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton CBF clustering -- pastikan sentiment_agg dihitung HANYA dari "
        "data train sebelum dipakai membangun fitur item (cegah leakage)."
    )
