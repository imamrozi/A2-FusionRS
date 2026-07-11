"""
prepare_tripadvisor_dataset.py

Flatten dataset TripAdvisor Hotel Reviews (Kaggle joebeachcapital/
hotel-reviews, arsip archive.zip berisi reviews.csv + offerings.csv)
menjadi CSV datar sesuai skema src/data_loader.py, lalu subsample
stratified (reuse scripts/make_subset.py::make_stratified_subset TANPA
DIUBAH) supaya ukurannya tractable diproses lokal/Colab.

Skema sumber (sudah diverifikasi langsung terhadap file riil, full scan
878.561 baris -- lihat plan): baca LANGSUNG dari dalam archive.zip (tanpa
extract penuh ke disk, hemat ~2GB ruang).

reviews.csv:
- ratings: STRING REPR DICT PYTHON (wajib ast.literal_eval, BUKAN JSON) --
  ambil ratings['overall'] -> stars.
- author: juga dict-repr string, field 'id' -> user_id. HILANG di ~8.8%
  baris -- baris ini dibuang (user_id wajib ada utk pipeline rekomendasi).
- id -> review_id, offering_id -> business_id, text -> text,
  date -> date (sudah format ISO YYYY-MM-DD, tidak perlu parsing tambahan).

offerings.csv (di-join on offering_id == id):
- hotel_class -> business_stars
- address (dict-repr string) field 'locality' -> business_city
- business_categories dibiarkan kosong (semua baris type=='hotel', tidak
  ada sub-kategori jelas di dataset ini).

Usage:
    python prepare_tripadvisor_dataset.py \
        --input data/raw/tripadvisor/archive.zip \
        --output data/raw/tripadvisor_hotel/tripadvisor_hotel_reviews.csv \
        --target-rows 120000
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.make_subset import make_stratified_subset  # noqa: E402
from src.data_loader import CORE_REQUIRED_COLUMNS, basic_clean_generic, filter_min_interactions_generic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)

OUTPUT_COLUMNS = CORE_REQUIRED_COLUMNS + ["business_stars", "business_city"]


def _safe_literal_eval(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def load_offerings_lookup(zf: zipfile.ZipFile) -> dict[str, dict]:
    """Bangun lookup offering_id -> {business_stars, business_city}."""
    lookup: dict[str, dict] = {}
    with zf.open("offerings.csv") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        reader = csv.DictReader(text)
        for row in reader:
            address = _safe_literal_eval(row.get("address"))
            lookup[row["id"]] = {
                "business_stars": row.get("hotel_class") or None,
                "business_city": address.get("locality"),
            }
    logger.info("offerings.csv dimuat: %d hotel", len(lookup))
    return lookup


def stream_flatten(input_path: Path, log_every: int = 100_000, max_rows: int | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(input_path) as zf:
        offerings_lookup = load_offerings_lookup(zf)

        rows: list[dict] = []
        n_read = 0
        n_skipped_missing_author = 0
        n_skipped_missing_field = 0

        with zf.open("reviews.csv") as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text)
            for row in reader:
                if max_rows is not None and n_read >= max_rows:
                    logger.info("Berhenti di batas --max-rows-preview=%d (mode validasi cepat)", max_rows)
                    break
                n_read += 1

                ratings = _safe_literal_eval(row.get("ratings"))
                stars = ratings.get("overall")

                author = _safe_literal_eval(row.get("author"))
                user_id = author.get("id")

                review_id = row.get("id")
                business_id = row.get("offering_id")
                review_text = row.get("text")
                date = row.get("date")

                if not user_id:
                    n_skipped_missing_author += 1
                    continue
                if not review_id or not business_id or not review_text or stars is None or not date:
                    n_skipped_missing_field += 1
                    continue

                biz = offerings_lookup.get(business_id, {})
                rows.append(
                    {
                        "review_id": review_id,
                        "user_id": user_id,
                        "business_id": business_id,
                        "text": review_text,
                        "stars": float(stars),
                        "date": date,
                        "business_stars": biz.get("business_stars"),
                        "business_city": biz.get("business_city"),
                    }
                )

                if n_read % log_every == 0:
                    logger.info("...sudah membaca %d baris mentah (%d valid)", n_read, len(rows))

    logger.info(
        "Selesai flatten: %d baris mentah dibaca, %d valid, %d dilewati (author/user_id kosong), "
        "%d dilewati (field wajib lain kosong)",
        n_read,
        len(rows),
        n_skipped_missing_author,
        n_skipped_missing_field,
    )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten + subsample dataset TripAdvisor hotel-reviews (archive.zip) menjadi CSV siap pipeline"
    )
    parser.add_argument("--input", type=str, required=True, help="Path archive.zip mentah")
    parser.add_argument("--output", type=str, required=True, help="Path output CSV")
    parser.add_argument("--target-rows", type=int, default=120_000, help="Target jumlah baris setelah subsampling")
    parser.add_argument("--min-reviews-per-user", type=int, default=5)
    parser.add_argument("--min-reviews-per-item", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows-preview",
        type=int,
        default=None,
        help="Batasi baris reviews.csv yang dibaca (utk validasi cepat sebelum full scan 878rb baris)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    logger.info("Flatten %s ...", input_path)
    df = stream_flatten(input_path, max_rows=args.max_rows_preview)

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
