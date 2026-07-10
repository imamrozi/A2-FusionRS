"""
run_significance_test.py

Uji signifikansi statistik (Wilcoxon signed-rank, default) antara DUA model
yang dibandingkan, berdasarkan file `predictions_*.csv` (squared error
per-sampel) yang disimpan run_baseline.py/run_baseline_absa.py/
run_classical_cf.py (lihat `save_predictions()` di src/evaluation/metrics.py).

WAJIB kedua model dievaluasi pada test set yang SAMA PERSIS (split identik)
-- ini dijamin oleh desain proyek (semua model pakai split yang sama), tapi
tetap divalidasi eksplisit lewat join `review_id` di sini (bukan asumsi
buta) -- kalau jumlah baris tak cocok setelah join, ada WARNING jelas.

Kalau `--seeds` diberi lebih dari 1 nilai, uji dijalankan TERPISAH per seed
(bukan digabung jadi 1 uji) -- tiap seed punya realisasi stokastik model
yang berbeda, jadi p-value per seed dilaporkan apa adanya. Latar belakang:
di percakapan sebelumnya, baseline (SA global) vs ABSA-concat cuma beda
0,3% RMSE pada 1 seed -- jauh lebih kecil dari variasi run-ke-run yang
teramati (~1,4%) pada config yang IDENTIK, sehingga perbandingan 1-seed
tidak cukup meyakinkan tanpa uji ini.

Usage:
    # Jalankan model A & B dengan seed tambahan dulu (tahap 5-8 saja, cepat
    # karena split/checkpoint/cache tahap 1-4 dipakai bersama):
    python run_baseline.py --config configs/yelp_config_colab.yaml --seed 123
    python run_baseline.py --config configs/yelp_config_colab.yaml --seed 456
    python run_baseline_absa.py --config configs/yelp_config_absa_concat_colab.yaml --seed 123
    python run_baseline_absa.py --config configs/yelp_config_absa_concat_colab.yaml --seed 456

    # Baru uji signifikansinya:
    python run_significance_test.py \
        --model-a baseline_reimpl --model-b absa_ablation_concat \
        --domain restaurant --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import significance_test

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_one_seed(
    results_dir: Path, model_a: str, model_b: str, domain: str, seed: int, test: str
) -> dict:
    path_a = results_dir / f"predictions_{model_a}_{domain}_seed{seed}.csv"
    path_b = results_dir / f"predictions_{model_b}_{domain}_seed{seed}.csv"

    if not path_a.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {path_a} -- jalankan model A dgn --seed {seed} dulu.")
    if not path_b.exists():
        raise FileNotFoundError(f"Tidak ditemukan: {path_b} -- jalankan model B dgn --seed {seed} dulu.")

    df_a = pd.read_csv(path_a)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_a"})
    df_b = pd.read_csv(path_b)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_b"})
    merged = df_a.merge(df_b, on="review_id", how="inner")

    if len(merged) != len(df_a) or len(merged) != len(df_b):
        logger.warning(
            "seed=%d: jumlah baris tak cocok setelah join review_id (model_a=%d, "
            "model_b=%d, terpasangkan=%d) -- kedua model mungkin TIDAK dievaluasi "
            "pada test set yang identik. Periksa split.output_dir di config masing-masing.",
            seed, len(df_a), len(df_b), len(merged),
        )

    statistic, p_value = significance_test(merged["se_a"].values, merged["se_b"].values, test=test)
    return {
        "seed": seed,
        "n_paired": len(merged),
        "rmse_a": float(merged["se_a"].mean() ** 0.5),
        "rmse_b": float(merged["se_b"].mean() ** 0.5),
        "statistic": statistic,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Uji signifikansi (Wilcoxon) antara 2 model")
    parser.add_argument("--model-a", type=str, required=True, help="Prefix file model A, mis. 'baseline_reimpl'")
    parser.add_argument("--model-b", type=str, required=True, help="Prefix file model B, mis. 'absa_ablation_concat'")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--test", type=str, default="wilcoxon", choices=["wilcoxon", "paired_t"])
    parser.add_argument("--results-dir", type=str, default="checkpoints/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for seed in args.seeds:
        try:
            row = run_one_seed(results_dir, args.model_a, args.model_b, args.domain, seed, args.test)
            rows.append(row)
            logger.info(
                "seed=%d: RMSE %s=%.4f vs %s=%.4f | %s statistic=%.2f, p_value=%.4f%s",
                seed, args.model_a, row["rmse_a"], args.model_b, row["rmse_b"],
                args.test, row["statistic"], row["p_value"],
                " (SIGNIFIKAN, p<0.05)" if row["p_value"] < 0.05 else " (TIDAK signifikan, p>=0.05)",
            )
        except FileNotFoundError as e:
            logger.warning("Lewati seed=%d: %s", seed, e)

    if not rows:
        raise SystemExit("Tidak ada seed yang berhasil diuji -- cek file predictions_*.csv tersedia dulu.")

    summary_df = pd.DataFrame(rows)
    logger.info("\n%s", summary_df.to_string(index=False))

    n_significant = int((summary_df["p_value"] < 0.05).sum())
    logger.info(
        "Ringkasan: %d/%d seed menunjukkan perbedaan SIGNIFIKAN (p<0.05) antara %s vs %s.",
        n_significant, len(summary_df), args.model_a, args.model_b,
    )

    output_path = results_dir / f"significance_{args.model_a}_vs_{args.model_b}_{args.domain}.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info("Ringkasan uji signifikansi disimpan ke %s", output_path)


if __name__ == "__main__":
    main()
