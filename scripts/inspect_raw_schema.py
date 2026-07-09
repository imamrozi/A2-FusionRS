"""
scripts/inspect_raw_schema.py

Jalankan skrip ini PERTAMA KALI setelah dataset diunduh, SEBELUM subsetting
atau menjalankan pipeline penuh. Tujuannya memverifikasi:
1. Format file (CSV vs JSON-lines) -- penting karena dataset asal Kaggle
   "Yelp Recruiting Competition" biasanya berformat JSON-per-baris
   (yelp_training_set_review.json), BUKAN CSV, meskipun nama file yang
   dirujuk baseline paper berakhiran .csv (kemungkinan data.world
   mengonversinya). Jika file Anda ternyata .json, gunakan flag --format json.
2. Kolom yang tersedia vs REQUIRED_COLUMNS yang diharapkan data_loader.py
3. Jumlah baris riil dan distribusi rating

Usage:
    python scripts/inspect_raw_schema.py --path data/raw/yelp_training_set_review.csv
    python scripts/inspect_raw_schema.py --path data/raw/yelp_training_set_review.json --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import REQUIRED_COLUMNS  # noqa: E402


def load_any(path: str, fmt: str, n_preview_rows: int = 200_000) -> pd.DataFrame:
    """Load CSV atau JSON-lines, dengan pembacaan dibatasi n_preview_rows
    baris pertama saja -- cukup untuk inspeksi skema tanpa harus memuat
    seluruh file (yang bisa berukuran ratusan MB-GB) ke memori."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    if fmt == "csv":
        return pd.read_csv(path, nrows=n_preview_rows)

    if fmt == "json":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_preview_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return pd.DataFrame(records)

    raise ValueError(f"Format '{fmt}' tidak dikenal, gunakan 'csv' atau 'json'")


def auto_detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    # coba deteksi dari isi baris pertama jika ekstensi ambigu
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    if first_line.startswith("{"):
        return "json"
    return "csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeksi skema dataset mentah")
    parser.add_argument("--path", type=str, required=True, help="Path ke file dataset mentah")
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "csv", "json"],
        help="Format file. 'auto' akan menebak dari ekstensi/isi file.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=200_000,
        help="Jumlah baris pertama yang dibaca untuk inspeksi (bukan seluruh file)",
    )
    args = parser.parse_args()

    fmt = auto_detect_format(args.path) if args.format == "auto" else args.format
    print(f"Format terdeteksi/dipilih: {fmt}")

    df = load_any(args.path, fmt, args.preview_rows)

    print("\n" + "=" * 70)
    print(f"JUMLAH BARIS (dari {args.preview_rows} baris pertama yang dibaca): {len(df)}")
    print("=" * 70)

    print("\n--- KOLOM YANG TERSEDIA ---")
    for col in df.columns:
        print(f"  - {col} (dtype: {df[col].dtype})")

    print("\n--- PERBANDINGAN DENGAN REQUIRED_COLUMNS (data_loader.py) ---")
    available = set(df.columns)
    required = set(REQUIRED_COLUMNS)
    missing = required - available
    extra = available - required

    if missing:
        print(f"  KOLOM WAJIB YANG HILANG: {sorted(missing)}")
        print(
            "  -> data_loader.py TIDAK akan bisa dipakai langsung. Perlu "
            "  mapping/rename kolom, atau sesuaikan REQUIRED_COLUMNS jika "
            "  kolom yang hilang memang tidak esensial untuk dataset ini."
        )
    else:
        print("  Semua kolom wajib TERSEDIA. data_loader.py bisa dipakai langsung.")

    if extra:
        print(f"\n  Kolom tambahan yang tersedia (tidak dipakai data_loader.py saat ini): {sorted(extra)}")

    print("\n--- 3 BARIS PERTAMA ---")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.head(3))

    if "stars" in df.columns:
        print("\n--- DISTRIBUSI RATING (stars) ---")
        print(df["stars"].value_counts().sort_index())
    elif "review_stars" in df.columns:
        print(
            "\n  Kolom rating kemungkinan bernama 'review_stars' bukan 'stars' -- "
            "  perlu rename sebelum dipakai data_loader.py."
        )

    if "business_categories" in df.columns:
        print("\n--- CONTOH NILAI business_categories (10 unik pertama) ---")
        print(df["business_categories"].dropna().unique()[:10])
    else:
        candidate_cols = [c for c in df.columns if "categor" in c.lower()]
        if candidate_cols:
            print(
                f"\n  Kolom kategori kemungkinan bernama salah satu dari: {candidate_cols} "
                "  (bukan 'business_categories') -- perlu rename."
            )

    print("\n" + "=" * 70)
    print("Selesai. Jika ada kolom hilang/nama berbeda, sesuaikan dulu di ")
    print("src/data_loader.py (REQUIRED_COLUMNS & referensi kolom) sebelum ")
    print("menjalankan scripts/make_subset.py atau run_baseline.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
