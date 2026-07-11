"""
src/data_loader.py

Modul pemuatan dataset mentah. Dirancang agar dataset loader terpisah dari
logic split dan preprocessing, sehingga bisa dipakai ulang baik oleh
reimplementasi baseline (Darraz et al.) maupun A2-FusionRS, dengan hasil
yang identik untuk menjamin fair comparison.

Mendukung multi-domain via LOADER_REGISTRY: "yelp" (restaurant/hotel,
format asli), "amazon" (e-commerce, hasil flatten prepare_amazon_dataset.py),
"tripadvisor" (hotel, skeleton -- menunggu dataset final terverifikasi).

STATUS: YelpDatasetLoader diuji pada data riil (domain restaurant, hasil
join review/business/user via prepare_yelp_dataset.py -- lihat README
bagian "Langkah Sebelum Menjalankan Eksperimen"). AmazonDatasetLoader diuji
pada data riil (domain electronics, hasil flatten prepare_amazon_dataset.py).
TripAdvisorDatasetLoader diuji pada data riil (domain hotel, hasil flatten
prepare_tripadvisor_dataset.py dari Kaggle joebeachcapital/hotel-reviews).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Kolom yang WAJIB ada di semua domain -- tanpa ini pipeline rekomendasi
# (split per-user, DeepMF, evaluasi) tidak bisa jalan sama sekali.
CORE_REQUIRED_COLUMNS = [
    "review_id",
    "user_id",
    "business_id",
    "text",
    "stars",
    "date",
]

# Kolom atribut bisnis/reviewer tambahan yang dipakai cbf_clustering.py utk
# fitur konten & popularitas. Opsional di level loader -- domain yang tidak
# punya metadata ini (mis. Amazon 5-core tanpa file metadata produk) tetap
# bisa dipakai; cbf_clustering.py otomatis mundur ke TF-IDF+numerik saja.
OPTIONAL_BUSINESS_COLUMNS = [
    "business_categories",
    "business_stars",
    "business_city",
    "business_review_count",
    "reviewer_review_count",
    "reviewer_average_stars",
]

# Dipertahankan utk backward-compat (dipakai scripts/inspect_raw_schema.py
# dan sebagai referensi skema Yelp Table 2 baseline paper). Nilai/urutan
# TIDAK berubah dari sebelum kolom opsional dipisah.
REQUIRED_COLUMNS = CORE_REQUIRED_COLUMNS + OPTIONAL_BUSINESS_COLUMNS


@dataclass
class DatasetStats:
    domain: str
    n_reviews: int
    n_users: int
    n_items: int
    sparsity: float
    rating_min: float
    rating_max: float


def ensure_optional_business_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Isi kolom atribut bisnis/reviewer opsional yang tidak tersedia dgn
    null, supaya tahap berikutnya (terutama cbf_clustering.py) tidak pernah
    KeyError, apapun domainnya. Loader domain baru WAJIB memanggil ini
    sebelum return dari load().
    """
    df = df.copy()
    for col in OPTIONAL_BUSINESS_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def basic_clean_generic(
    df: pd.DataFrame, text_col: str, rating_col: str, user_col: str, item_col: str
) -> pd.DataFrame:
    """Versi generik dari YelpDatasetLoader._basic_clean -- buang baris
    null/rating di luar 1-5, dgn nama kolom yang bisa disesuaikan per
    domain (semua domain saat ini memakai skema kolom sama, tapi
    dipisah agar eksplisit & reusable)."""
    before = len(df)
    df = df.dropna(subset=[text_col, rating_col, user_col, item_col])
    df = df[df[text_col].str.strip().str.len() > 0]
    df[rating_col] = df[rating_col].astype(float)
    df = df[df[rating_col].between(1, 5)]
    after = len(df)
    if after < before:
        logger.info("Membuang %d baris tidak valid (null/rating di luar 1-5)", before - after)
    return df.reset_index(drop=True)


def filter_min_interactions_generic(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    min_reviews_per_user: int,
    min_reviews_per_item: int,
) -> pd.DataFrame:
    """Versi generik dari YelpDatasetLoader.filter_min_interactions.

    Dilakukan iteratif (bukan sekali filter) karena membuang user dengan
    interaksi rendah bisa membuat sejumlah item jatuh di bawah threshold,
    dan sebaliknya -- perlu diulang sampai stabil.
    """
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        valid_users = user_counts[user_counts >= min_reviews_per_user].index
        valid_items = item_counts[item_counts >= min_reviews_per_item].index
        df = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]
    return df.reset_index(drop=True)


def compute_stats_generic(
    df: pd.DataFrame, domain: str, user_col: str, item_col: str, rating_col: str
) -> DatasetStats:
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_reviews = len(df)
    sparsity = 1 - (n_reviews / (n_users * n_items)) if n_users and n_items else float("nan")
    return DatasetStats(
        domain=domain,
        n_reviews=n_reviews,
        n_users=n_users,
        n_items=n_items,
        sparsity=sparsity,
        rating_min=df[rating_col].min(),
        rating_max=df[rating_col].max(),
    )


class YelpDatasetLoader:
    """Loader untuk dataset review Yelp (domain restaurant/hotel).

    Parameters
    ----------
    raw_path : str | Path
        Path ke file csv mentah (hasil unduhan manual dari data.world).
    domain : str
        "restaurant" atau "hotel" -- dipakai untuk memfilter business_categories.
    """

    # Kata kunci kategori bisnis untuk filtering domain. Perlu disesuaikan
    # setelah inspeksi nilai unik `business_categories` pada dataset riil --
    # daftar di bawah adalah asumsi awal berbasis deskripsi umum Yelp dataset,
    # BUKAN hasil verifikasi langsung terhadap file.
    DOMAIN_KEYWORDS = {
        "restaurant": ["restaurant", "food", "cafe", "diner", "bar"],
        "hotel": ["hotel", "motel", "resort", "lodging", "inn"],
    }

    def __init__(self, raw_path: str | Path, domain: str = "restaurant"):
        self.raw_path = Path(raw_path)
        if domain not in self.DOMAIN_KEYWORDS:
            raise ValueError(
                f"domain harus salah satu dari {list(self.DOMAIN_KEYWORDS)}, dapat '{domain}'"
            )
        self.domain = domain

    def load(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {self.raw_path}. "
                "Unduh manual dari data.world/brianray/yelp-reviews dan "
                "letakkan sesuai path di configs/yelp_config.yaml."
            )

        logger.info("Memuat dataset dari %s", self.raw_path)
        df = pd.read_csv(self.raw_path)
        self.validate_schema(df)

        df = self._filter_domain(df)
        df = basic_clean_generic(df, "text", "stars", "user_id", "business_id")

        logger.info(
            "Dataset domain '%s' dimuat: %d baris setelah filtering",
            self.domain,
            len(df),
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Verifikasi kolom wajib tersedia sebelum diproses lebih jauh.

        Dipanggil otomatis oleh `load()`, tapi bisa dipanggil manual saat
        eksplorasi awal file dataset (misal di notebooks/01_eda.ipynb) untuk
        memastikan skema file yang diunduh sesuai ekspektasi.
        """
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan pada dataset: {sorted(missing)}. "
                "Periksa apakah file yang diunduh sesuai dengan versi yang "
                "digunakan baseline paper (Darraz et al. 2025), karena skema "
                "data.world bisa berubah antar snapshot."
            )

    def _filter_domain(self, df: pd.DataFrame) -> pd.DataFrame:
        keywords = self.DOMAIN_KEYWORDS[self.domain]
        pattern = "|".join(keywords)
        mask = df["business_categories"].str.contains(
            pattern, case=False, na=False, regex=True
        )
        filtered = df[mask].copy()
        if filtered.empty:
            raise ValueError(
                f"Filtering domain '{self.domain}' menghasilkan 0 baris. "
                "Cek nilai unik df['business_categories'] secara manual -- "
                "kemungkinan keyword tidak cocok dengan format kategori riil."
            )
        return filtered

    def filter_min_interactions(
        self, df: pd.DataFrame, min_reviews_per_user: int, min_reviews_per_item: int
    ) -> pd.DataFrame:
        return filter_min_interactions_generic(
            df, "user_id", "business_id", min_reviews_per_user, min_reviews_per_item
        )

    def compute_stats(self, df: pd.DataFrame) -> DatasetStats:
        return compute_stats_generic(df, self.domain, "user_id", "business_id", "stars")


class AmazonDatasetLoader:
    """Loader untuk dataset review Amazon (format hasil flatten
    prepare_amazon_dataset.py dari file .json.gz "5-core" McAuley Lab).

    Tidak ada filtering domain via business_categories (tidak tersedia --
    file metadata produk terpisah tidak ada di sumber data ini) -- kategori
    sudah ditentukan oleh file input itu sendiri (mis. satu file per
    kategori Electronics/Clothing/Beauty), jadi `domain` di sini murni
    label deskriptif (dipakai utk penamaan hasil & pemilihan
    DEFAULT_ASPECT_KEYWORDS di absa_bert.py), BUKAN kriteria filter.

    Parameters
    ----------
    raw_path : str | Path
        Path ke CSV hasil prepare_amazon_dataset.py (bukan .json.gz mentah).
    domain : str
        Label domain, mis. "amazon_electronics".
    """

    def __init__(self, raw_path: str | Path, domain: str = "amazon_electronics"):
        self.raw_path = Path(raw_path)
        self.domain = domain

    def load(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {self.raw_path}. "
                "Jalankan prepare_amazon_dataset.py dulu untuk menghasilkan "
                "CSV flatten dari file .json.gz mentah."
            )

        logger.info("Memuat dataset Amazon dari %s", self.raw_path)
        df = pd.read_csv(self.raw_path)
        self.validate_schema(df)
        df = ensure_optional_business_columns(df)

        df = basic_clean_generic(df, "text", "stars", "user_id", "business_id")

        logger.info(
            "Dataset Amazon domain '%s' dimuat: %d baris setelah cleaning",
            self.domain,
            len(df),
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = set(CORE_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib (CORE_REQUIRED_COLUMNS) tidak ditemukan pada dataset Amazon: "
                f"{sorted(missing)}. Periksa hasil prepare_amazon_dataset.py."
            )

    def filter_min_interactions(
        self, df: pd.DataFrame, min_reviews_per_user: int, min_reviews_per_item: int
    ) -> pd.DataFrame:
        return filter_min_interactions_generic(
            df, "user_id", "business_id", min_reviews_per_user, min_reviews_per_item
        )

    def compute_stats(self, df: pd.DataFrame) -> DatasetStats:
        return compute_stats_generic(df, self.domain, "user_id", "business_id", "stars")


class TripAdvisorDatasetLoader:
    """Loader untuk dataset review hotel TripAdvisor (Kaggle
    joebeachcapital/hotel-reviews, format hasil flatten
    prepare_tripadvisor_dataset.py dari reviews.csv+offerings.csv di
    dalam archive.zip).

    Skema sumber sudah full-scan-verified (878.561 baris) -- lihat
    docstring prepare_tripadvisor_dataset.py. `business_stars` (dari
    hotel_class) dan `business_city` (dari address.locality) sudah
    di-join di tahap flatten, jadi biasanya sudah terisi (bukan null)
    utk sebagian besar baris -- beda dgn AmazonDatasetLoader yang sama
    sekali tidak punya metadata bisnis.

    Tidak ada filtering domain (semua offerings di dataset sumber sudah
    type=='hotel') -- `domain` di sini murni label deskriptif, sama
    seperti AmazonDatasetLoader.

    Parameters
    ----------
    raw_path : str | Path
        Path ke CSV hasil prepare_tripadvisor_dataset.py (bukan archive.zip mentah).
    domain : str
        Label domain, mis. "tripadvisor_hotel".
    """

    def __init__(self, raw_path: str | Path, domain: str = "tripadvisor_hotel"):
        self.raw_path = Path(raw_path)
        self.domain = domain

    def load(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di {self.raw_path}. "
                "Jalankan prepare_tripadvisor_dataset.py dulu untuk menghasilkan "
                "CSV flatten dari data/raw/tripadvisor/archive.zip."
            )

        logger.info("Memuat dataset TripAdvisor dari %s", self.raw_path)
        df = pd.read_csv(self.raw_path)
        self.validate_schema(df)
        df = ensure_optional_business_columns(df)

        df = basic_clean_generic(df, "text", "stars", "user_id", "business_id")

        logger.info(
            "Dataset TripAdvisor domain '%s' dimuat: %d baris setelah cleaning",
            self.domain,
            len(df),
        )
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = set(CORE_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib (CORE_REQUIRED_COLUMNS) tidak ditemukan pada dataset TripAdvisor: "
                f"{sorted(missing)}. Periksa hasil prepare_tripadvisor_dataset.py."
            )

    def filter_min_interactions(
        self, df: pd.DataFrame, min_reviews_per_user: int, min_reviews_per_item: int
    ) -> pd.DataFrame:
        return filter_min_interactions_generic(
            df, "user_id", "business_id", min_reviews_per_user, min_reviews_per_item
        )

    def compute_stats(self, df: pd.DataFrame) -> DatasetStats:
        return compute_stats_generic(df, self.domain, "user_id", "business_id", "stars")


LOADER_REGISTRY = {
    "yelp": YelpDatasetLoader,
    "amazon": AmazonDatasetLoader,
    "tripadvisor": TripAdvisorDatasetLoader,
}


def get_loader_class(name: str):
    if name not in LOADER_REGISTRY:
        raise ValueError(
            f"Loader '{name}' tidak dikenal. Pilihan tersedia: {sorted(LOADER_REGISTRY)}"
        )
    return LOADER_REGISTRY[name]


if __name__ == "__main__":
    # Contoh pemakaian cepat untuk verifikasi awal - jalankan manual setelah
    # dataset diunduh, sebelum menjalankan pipeline penuh.
    logging.basicConfig(level=logging.INFO)
    loader = YelpDatasetLoader("data/raw/yelp_training_set_review.csv", domain="restaurant")
    data = loader.load()
    data = loader.filter_min_interactions(data, min_reviews_per_user=5, min_reviews_per_item=5)
    stats = loader.compute_stats(data)
    logger.info(stats)
