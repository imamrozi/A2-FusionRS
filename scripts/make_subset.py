"""
scripts/make_subset.py

Membuat subset kecil (default ~5.000 baris) dari dataset mentah untuk
validasi cepat pipeline end-to-end SEBELUM full run yang memakan compute
budget Colab. Subset dibuat dengan stratifikasi sederhana: memprioritaskan
user dan item dengan interaksi cukup banyak, agar subset tidak didominasi
oleh cold-start user/item tunggal (yang membuat split_generator.py
langsung memasukkan hampir semua baris ke cold_start set, bukan
train/val/test -- sehingga smoke test jadi tidak representatif).

PENTING: Jalankan scripts/inspect_raw_schema.py TERLEBIH DAHULU untuk
memastikan path & format file benar, serta kolom-kolom sesuai ekspektasi
data_loader.py, sebelum menjalankan script ini.

Usage:
    python scripts/make_subset.py \\
        --input data/raw/yelp_training_set_review.csv \\
        --output data/raw/yelp_subset_5k.csv \\
        --domain restaurant \\
        --n-rows 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import YelpDatasetLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def make_stratified_subset(
    df: pd.DataFrame,
    n_rows: int,
    min_reviews_per_user: int = 5,
    min_reviews_per_item: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Ambil subset dengan strategi:
    1. Filter dulu user/item dengan interaksi >= threshold (sama seperti
       yang akan dilakukan filter_min_interactions() pada pipeline penuh)
       -- ini membuat subset REPRESENTATIF terhadap kondisi yang akan
       dihadapi split_generator.py nanti (bukan subset acak murni yang
       bisa didominasi user dengan 1 interaksi).
    2. Ambil user secara acak (bukan baris acak) sampai jumlah baris
       terkumpul mendekati n_rows -- ini menjaga agar tiap user yang masuk
       subset punya cukup banyak interaksinya sendiri (tidak terpotong),
       lebih representatif untuk menguji user-based split.
    """
    rng = np.random.default_rng(seed)

    user_counts = df["user_id"].value_counts()
    item_counts = df["business_id"].value_counts()
    valid_users = user_counts[user_counts >= min_reviews_per_user].index
    valid_items = item_counts[item_counts >= min_reviews_per_item].index

    filtered = df[df["user_id"].isin(valid_users) & df["business_id"].isin(valid_items)]
    logger.info(
        "Setelah filter min_reviews (user>=%d, item>=%d): %d baris tersisa dari %d baris awal",
        min_reviews_per_user,
        min_reviews_per_item,
        len(filtered),
        len(df),
    )

    if len(filtered) <= n_rows:
        logger.warning(
            "Baris hasil filter (%d) sudah <= target n_rows (%d) -- subset "
            "akan memakai semua baris hasil filter tanpa sampling tambahan. "
            "Pertimbangkan menurunkan min_reviews_per_user/item jika subset "
            "terlalu kecil untuk smoke test yang bermakna.",
            len(filtered),
            n_rows,
        )
        return filtered.reset_index(drop=True)

    shuffled_users = rng.permutation(filtered["user_id"].unique())
    selected_rows = []
    total = 0

    for user_id in shuffled_users:
        user_rows = filtered[filtered["user_id"] == user_id]
        selected_rows.append(user_rows)
        total += len(user_rows)
        if total >= n_rows:
            break

    subset = pd.concat(selected_rows).reset_index(drop=True)
    logger.info("Subset akhir: %d baris, %d user unik, %d item unik",
                len(subset), subset["user_id"].nunique(), subset["business_id"].nunique())
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat subset dataset untuk smoke test pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path dataset mentah (hasil unduhan)")
    parser.add_argument("--output", type=str, required=True, help="Path output subset (.csv)")
    parser.add_argument(
        "--domain", type=str, default="restaurant", choices=["restaurant", "hotel"],
        help="Domain yang difilter, sesuai src/data_loader.py",
    )
    parser.add_argument("--n-rows", type=int, default=5000, help="Target jumlah baris subset")
    parser.add_argument("--min-reviews-per-user", type=int, default=5)
    parser.add_argument("--min-reviews-per-item", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Memuat & memfilter domain '%s' dari %s ...", args.domain, args.input)
    loader = YelpDatasetLoader(args.input, domain=args.domain)
    df = loader.load()

    logger.info("Membuat subset stratified (target %d baris)...", args.n_rows)
    subset = make_stratified_subset(
        df,
        n_rows=args.n_rows,
        min_reviews_per_user=args.min_reviews_per_user,
        min_reviews_per_item=args.min_reviews_per_item,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)

    logger.info("Subset disimpan ke %s", output_path)
    logger.info(
        "Selanjutnya: jalankan pipeline dengan config yang menunjuk ke file "
        "ini (lihat configs/yelp_config_quicktest.yaml) untuk validasi "
        "end-to-end sebelum full run."
    )


if __name__ == "__main__":
    main()
