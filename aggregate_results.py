"""
aggregate_results.py

Baca SEMUA file hasil eksperimen (results/*.yaml) yang dihasilkan
run_baseline.py, run_baseline_absa.py, dan run_classical_cf.py -- lintas
skenario, domain, dan seed -- lalu susun jadi satu tabel perbandingan siap
ditempel ke manuskrip (CSV, mean +/- std kalau ada >1 seed per model+domain).

Pakai aggregate_multi_seed_results() dari src/evaluation/metrics.py (sudah
ada sebelumnya tapi belum pernah dipanggil dari mana pun).

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir checkpoints/results --output checkpoints/results/comparison_table.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.metrics import RunResult, aggregate_multi_seed_results

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_dir: str | Path) -> list[RunResult]:
    """Scan semua *.yaml (BUKAN predictions_*.csv) di results_dir, parse jadi
    list RunResult. File yang gagal di-parse (mis. field wajib hilang)
    di-skip dengan warning, bukan menghentikan seluruh agregasi."""
    results: list[RunResult] = []
    yaml_paths = sorted(Path(results_dir).glob("*.yaml"))
    for path in yaml_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            results.append(
                RunResult(
                    model_name=data["model_name"],
                    domain=data.get("domain", ""),
                    seed=data["seed"],
                    rmse=data["rmse"],
                    mae=data["mae"],
                    precision_at_k=data.get("precision_at_k", {}) or {},
                    recall_at_k=data.get("recall_at_k", {}) or {},
                    ndcg_at_k=data.get("ndcg_at_k", {}) or {},
                )
            )
        except (KeyError, TypeError) as e:
            logger.warning("Lewati %s -- field wajib hilang/rusak (%s)", path, e)
    logger.info("Berhasil memuat %d/%d file hasil dari %s", len(results), len(yaml_paths), results_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agregasi semua hasil eksperimen (results/*.yaml) jadi 1 tabel perbandingan"
    )
    parser.add_argument("--results-dir", type=str, default="checkpoints/results")
    parser.add_argument(
        "--output", type=str, default="checkpoints/results/comparison_table.csv"
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(
            f"Tidak ada file *.yaml valid ditemukan di {args.results_dir} -- "
            "jalankan run_baseline.py/run_baseline_absa.py/run_classical_cf.py dulu."
        )

    summary = aggregate_multi_seed_results(results)
    # Ratakan kolom multi-index (metric, stat) -> "metric_stat" supaya CSV-nya
    # 1 baris header, langsung enak dibaca/ditempel ke manuskrip.
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary)
    logger.info("Tabel perbandingan (%d baris) disimpan ke %s", len(summary), args.output)


if __name__ == "__main__":
    main()
