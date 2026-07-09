"""
prepare_yelp_dataset.py

Menggabungkan file mentah dataset "Yelp Training Set" (format lama Yelp
Recruiting Competition di data.world/brianray/yelp-reviews: file JSON
terpisah per entitas) menjadi satu CSV datar sesuai skema yang diharapkan
`src/data_loader.py::REQUIRED_COLUMNS`.

review.json  -> review_id, user_id, business_id, text, stars, date
business.json -> business_categories, business_stars, business_city,
                  business_review_count (join by business_id)
user.json    -> reviewer_review_count, reviewer_average_stars
                  (join by user_id)

Usage:
    python prepare_yelp_dataset.py \
        --input-dir data/raw/yelp_training_set_all \
        --output data/raw/yelp_training_set_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
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


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_lookup(records: list[dict], key: str) -> dict[str, dict]:
    return {r[key]: r for r in records}


def merge(input_dir: Path, output_path: Path) -> None:
    business_path = input_dir / "yelp_training_set_business.json"
    user_path = input_dir / "yelp_training_set_user.json"
    review_path = input_dir / "yelp_training_set_review.json"

    for p in (business_path, user_path, review_path):
        if not p.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    logger.info("Memuat business.json ...")
    business_lookup = build_lookup(_load_jsonl(business_path), "business_id")
    logger.info("Memuat user.json ...")
    user_lookup = build_lookup(_load_jsonl(user_path), "user_id")

    logger.info("Menggabungkan review.json dengan business & user (streaming) ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_business = 0
    n_skipped_user = 0

    with open(review_path, encoding="utf-8") as fin, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for line in fin:
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)

            biz = business_lookup.get(review["business_id"])
            if biz is None:
                n_skipped_business += 1
                continue

            user = user_lookup.get(review["user_id"])
            if user is None:
                n_skipped_user += 1
                continue

            writer.writerow(
                {
                    "review_id": review["review_id"],
                    "user_id": review["user_id"],
                    "business_id": review["business_id"],
                    "text": review["text"],
                    "stars": review["stars"],
                    "date": review["date"],
                    "business_categories": "|".join(biz.get("categories") or []),
                    "business_stars": biz.get("stars"),
                    "business_city": biz.get("city"),
                    "business_review_count": biz.get("review_count"),
                    "reviewer_review_count": user.get("review_count"),
                    "reviewer_average_stars": user.get("average_stars"),
                }
            )
            n_written += 1

    logger.info(
        "Selesai: %d baris ditulis ke %s (dilewati: %d tanpa business match, "
        "%d tanpa user match)",
        n_written,
        output_path,
        n_skipped_business,
        n_skipped_user,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gabungkan JSON review/business/user Yelp menjadi satu CSV"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/yelp_training_set_all",
        help="Folder berisi yelp_training_set_{business,user,review}.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/yelp_training_set_review.csv",
        help="Path output CSV (harus sesuai data.raw_path di config)",
    )
    args = parser.parse_args()

    merge(Path(args.input_dir), Path(args.output))
