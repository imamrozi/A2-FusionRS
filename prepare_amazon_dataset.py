"""
prepare_amazon_dataset.py

Flatten file mentah Amazon Review Data "5-core" (McAuley Lab,
format .json.gz, satu record JSON per baris) menjadi CSV datar sesuai
CORE_REQUIRED_COLUMNS di src/data_loader.py, lalu subsample stratified
(reuse scripts/make_subset.py::make_stratified_subset TANPA DIUBAH) supaya
ukurannya tractable diproses lokal/Colab.

Skema sumber (sudah diverifikasi langsung terhadap file riil -- lihat
plan): overall->stars, reviewerID->user_id, asin->business_id,
reviewText->text, unixReviewTime->date. TIDAK ada file metadata produk
terpisah (kategori/brand) -- kolom bisnis opsional diisi null oleh
AmazonDatasetLoader.load() via ensure_optional_business_columns(), bukan
di sini.

review_id disintesis (reviewerID::asin::unixReviewTime, + disambiguator
angka kalau ada tabrakan -- bisa terjadi kalau user mereview item yang
sama persis di detik unix time yang sama, jarang tapi mungkin).

Usage:
    python prepare_amazon_dataset.py \
        --input data/raw/amazon_beauty/All_Beauty_5.json.gz \
        --output data/raw/amazon_beauty/amazon_beauty_reviews.csv \
        --target-rows 5000

    python prepare_amazon_dataset.py \
        --input data/raw/amazon_electronics/Electronics_5.json.gz \
        --output data/raw/amazon_electronics/amazon_electronics_reviews.csv \
        --target-rows 120000
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.make_subset import make_stratified_subset  # noqa: E402
from src.data_loader import CORE_REQUIRED_COLUMNS, basic_clean_generic, filter_min_interactions_generic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def stream_flatten(input_path: Path, log_every: int = 500_000) -> pd.DataFrame:
    """Baca .json.gz baris-per-baris (streaming, hemat memori dibanding
    json.load() sekaligus), map ke skema CORE_REQUIRED_COLUMNS."""
    rows: list[dict] = []
    id_counter: collections.Counter = collections.Counter()
    n_read = 0
    n_skipped_missing_field = 0

    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            rec = json.loads(line)

            text = rec.get("reviewText")
            overall = rec.get("overall")
            reviewer_id = rec.get("reviewerID")
            asin = rec.get("asin")
            unix_time = rec.get("unixReviewTime")

            if not text or overall is None or not reviewer_id or not asin or unix_time is None:
                n_skipped_missing_field += 1
                continue

            base_id = f"{reviewer_id}::{asin}::{unix_time}"
            id_counter[base_id] += 1
            review_id = base_id if id_counter[base_id] == 1 else f"{base_id}::{id_counter[base_id]}"
            date = datetime.fromtimestamp(int(unix_time), tz=timezone.utc).strftime("%Y-%m-%d")

            rows.append(
                {
                    "review_id": review_id,
                    "user_id": reviewer_id,
                    "business_id": asin,
                    "text": text,
                    "stars": float(overall),
                    "date": date,
                }
            )

            if n_read % log_every == 0:
                logger.info("...sudah membaca %d baris mentah (%d valid)", n_read, len(rows))

    logger.info(
        "Selesai flatten: %d baris mentah dibaca, %d valid, %d dilewati (field wajib kosong)",
        n_read,
        len(rows),
        n_skipped_missing_field,
    )
    return pd.DataFrame(rows, columns=CORE_REQUIRED_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten + subsample dataset Amazon 5-core (.json.gz) menjadi CSV siap pipeline"
    )
    parser.add_argument("--input", type=str, required=True, help="Path file .json.gz mentah")
    parser.add_argument("--output", type=str, required=True, help="Path output CSV")
    parser.add_argument("--target-rows", type=int, default=120_000, help="Target jumlah baris setelah subsampling")
    parser.add_argument("--min-reviews-per-user", type=int, default=5)
    parser.add_argument("--min-reviews-per-item", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    logger.info("Flatten %s ...", input_path)
    df = stream_flatten(input_path)

    logger.info("Cleaning dasar (null/rating di luar 1-5)...")
    df = basic_clean_generic(df, "text", "stars", "user_id", "business_id")

    logger.info(
        "Filter interaksi minimum (user>=%d, item>=%d) sebelum sampling...",
        args.min_reviews_per_user,
        args.min_reviews_per_item,
    )
    df = filter_min_interactions_generic(
        df, "user_id", "business_id", args.min_reviews_per_user, args.min_reviews_per_item
    )
    logger.info("Baris tersisa setelah filter interaksi minimum: %d", len(df))

    if len(df) > args.target_rows:
        logger.info("Subsampling stratified ke target %d baris...", args.target_rows)
        # make_stratified_subset() sudah re-filter min_reviews_per_user/item
        # SENDIRI setelah sampling (lihat scripts/make_subset.py) -- tidak
        # perlu diulang di sini lagi.
        df = make_stratified_subset(
            df,
            n_rows=args.target_rows,
            min_reviews_per_user=args.min_reviews_per_user,
            min_reviews_per_item=args.min_reviews_per_item,
            seed=args.seed,
        )
    else:
        logger.info(
            "Baris hasil filter (%d) sudah <= target (%d) -- tidak ada subsampling tambahan.",
            len(df),
            args.target_rows,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Selesai: %d baris, %d user unik, %d item unik disimpan ke %s",
        len(df),
        df["user_id"].nunique(),
        df["business_id"].nunique(),
        output_path,
    )


if __name__ == "__main__":
    main()
