"""
src/data_loader.py

Modul pemuatan dataset mentah. Dirancang agar dataset loader terpisah dari
logic split dan preprocessing, sehingga bisa dipakai ulang baik oleh
reimplementasi baseline (Darraz et al.) maupun A2-FusionRS, dengan hasil
yang identik untuk menjamin fair comparison.

STATUS: diuji pada data riil (domain restaurant, hasil join review/business/
user via prepare_yelp_dataset.py -- lihat README bagian "Langkah Sebelum
Menjalankan Eksperimen"). Domain "hotel" dan skema dataset final untuk
manuskrip tetap perlu divalidasi ulang -- lihat `validate_schema()` di
bawah untuk bantuan verifikasi cepat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Kolom minimal yang diharapkan ada pada dataset Yelp, berdasarkan Table 2
# baseline paper (Darraz et al., 2025). Jika kolom ini tidak lengkap, loader
# akan raise error eksplisit -- lebih baik gagal cepat daripada diam-diam
# menghasilkan fitur kosong di tahap berikutnya.
REQUIRED_COLUMNS = [
    "review_id",
    "user_id",
    "business_id",
    "text",
    "stars",
    "date",
    "business_categories",
    "business_stars",
    "business_city",
    "business_review_count",
    "reviewer_review_count",
    "reviewer_average_stars",
]


@dataclass
class DatasetStats:
    domain: str
    n_reviews: int
    n_users: int
    n_items: int
    sparsity: float
    rating_min: float
    rating_max: float


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
        df = self._basic_clean(df)

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

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["text", "stars", "user_id", "business_id"])
        df = df[df["text"].str.strip().str.len() > 0]
        df["stars"] = df["stars"].astype(float)
        df = df[df["stars"].between(1, 5)]
        after = len(df)
        if after < before:
            logger.info("Membuang %d baris tidak valid (null/rating di luar 1-5)", before - after)
        return df.reset_index(drop=True)

    def filter_min_interactions(
        self, df: pd.DataFrame, min_reviews_per_user: int, min_reviews_per_item: int
    ) -> pd.DataFrame:
        """Iteratif filter user/item dengan interaksi minimum.

        Dilakukan iteratif (bukan sekali filter) karena membuang user dengan
        interaksi rendah bisa membuat sejumlah item jatuh di bawah threshold,
        dan sebaliknya -- perlu diulang sampai stabil.
        """
        prev_len = -1
        while len(df) != prev_len:
            prev_len = len(df)
            user_counts = df["user_id"].value_counts()
            item_counts = df["business_id"].value_counts()
            valid_users = user_counts[user_counts >= min_reviews_per_user].index
            valid_items = item_counts[item_counts >= min_reviews_per_item].index
            df = df[
                df["user_id"].isin(valid_users) & df["business_id"].isin(valid_items)
            ]
        return df.reset_index(drop=True)

    def compute_stats(self, df: pd.DataFrame) -> DatasetStats:
        n_users = df["user_id"].nunique()
        n_items = df["business_id"].nunique()
        n_reviews = len(df)
        sparsity = 1 - (n_reviews / (n_users * n_items)) if n_users and n_items else float("nan")
        return DatasetStats(
            domain=self.domain,
            n_reviews=n_reviews,
            n_users=n_users,
            n_items=n_items,
            sparsity=sparsity,
            rating_min=df["stars"].min(),
            rating_max=df["stars"].max(),
        )


if __name__ == "__main__":
    # Contoh pemakaian cepat untuk verifikasi awal - jalankan manual setelah
    # dataset diunduh, sebelum menjalankan pipeline penuh.
    logging.basicConfig(level=logging.INFO)
    loader = YelpDatasetLoader("data/raw/yelp_training_set_review.csv", domain="restaurant")
    data = loader.load()
    data = loader.filter_min_interactions(data, min_reviews_per_user=5, min_reviews_per_item=5)
    stats = loader.compute_stats(data)
    logger.info(stats)
