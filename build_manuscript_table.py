"""
build_manuscript_table.py

Gabungkan hasil aggregate_results.py (mean +/- std lintas-seed) dan
run_significance_test.py (uji Wilcoxon berpasangan per-seed vs model
referensi) jadi SATU tabel ringkasan -- 1 baris per model, siap ditempel
ke manuskrip.

Kolom signifikansi:
- n_significant_seeds: "k/N" -- jumlah seed dengan p<0.05 dari N seed yang
  berhasil diuji. Ini bukti PRIMER (paling mudah diinterpretasi & robust).
- p_value_combined_fisher: p-value gabungan lintas-seed (metode Fisher,
  scipy.stats.combine_pvalues) -- bukti SEKUNDER/pelengkap. CATATAN
  METODOLOGIS: metode ini idealnya untuk p-value dari uji yang independen;
  di sini ke-N uji per-seed memakai TEST SET yang SAMA (split identik
  antar-seed, cuma realisasi stokastik model yang beda per-seed) -- jadi
  bukan independensi sempurna secara teoretis, tapi tetap jadi ringkasan
  yang wajar & lazim dipakai di riset ML utk gabungkan bukti lintas-seed.
  WAJIB dilaporkan BERSAMA n_significant_seeds di manuskrip, bukan sendirian.

Usage:
    python build_manuscript_table.py \
        --reference baseline_reimpl \
        --models absa_ablation absa_ablation_confidence_mean \
                 absa_ablation_concat absa_ablation_concat_confidence \
        --domain restaurant --seeds 42 123 456 789 1011
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import combine_pvalues

from src.evaluation.metrics import RunResult, aggregate_multi_seed_results, significance_test

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_one_result(results_dir: Path, prefix: str, domain: str, seed: int) -> RunResult | None:
    path = results_dir / f"{prefix}_{domain}_seed{seed}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunResult(
        model_name=data["model_name"],
        domain=data.get("domain", ""),
        seed=data["seed"],
        rmse=data["rmse"],
        mae=data["mae"],
        precision_at_k=data.get("precision_at_k", {}) or {},
        recall_at_k=data.get("recall_at_k", {}) or {},
        ndcg_at_k=data.get("ndcg_at_k", {}) or {},
    )


def paired_p_value(results_dir: Path, prefix_a: str, prefix_b: str, domain: str, seed: int) -> float | None:
    path_a = results_dir / f"predictions_{prefix_a}_{domain}_seed{seed}.csv"
    path_b = results_dir / f"predictions_{prefix_b}_{domain}_seed{seed}.csv"
    if not path_a.exists() or not path_b.exists():
        return None
    df_a = pd.read_csv(path_a)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_a"})
    df_b = pd.read_csv(path_b)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_b"})
    merged = df_a.merge(df_b, on="review_id", how="inner")
    if merged.empty:
        return None
    _, p_value = significance_test(merged["se_a"].values, merged["se_b"].values, test="wilcoxon")
    return p_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bangun 1 tabel ringkasan manuskrip (mean+-std + signifikansi vs referensi)"
    )
    parser.add_argument("--reference", type=str, required=True, help="Prefix model referensi, mis. 'baseline_reimpl'")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Prefix model lain yang dibandingkan")
    parser.add_argument("--domain", type=str, default="restaurant")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    parser.add_argument("--results-dir", type=str, default="checkpoints/results")
    parser.add_argument("--output", type=str, default="checkpoints/results/manuscript_table.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_prefixes = [args.reference] + args.models

    rows = []
    for prefix in all_prefixes:
        run_results = []
        for seed in args.seeds:
            r = load_one_result(results_dir, prefix, args.domain, seed)
            if r is not None:
                run_results.append(r)
            else:
                logger.warning("Lewati %s seed=%d: file hasil tidak ditemukan.", prefix, seed)

        if not run_results:
            logger.warning("Lewati model %s -- tidak ada hasil ditemukan sama sekali.", prefix)
            continue

        summary = aggregate_multi_seed_results(run_results)
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        row = summary.iloc[0].to_dict()
        row["model_prefix"] = prefix
        row["n_seeds"] = len(run_results)

        if prefix != args.reference:
            p_values = []
            n_significant = 0
            for seed in args.seeds:
                p = paired_p_value(results_dir, args.reference, prefix, args.domain, seed)
                if p is not None:
                    p_values.append(p)
                    if p < 0.05:
                        n_significant += 1
            if p_values:
                _, combined_p = combine_pvalues(p_values, method="fisher")
                row["p_value_combined_fisher"] = combined_p
                row["n_significant_seeds"] = f"{n_significant}/{len(p_values)}"
            else:
                row["p_value_combined_fisher"] = None
                row["n_significant_seeds"] = "0/0"
        else:
            row["p_value_combined_fisher"] = None
            row["n_significant_seeds"] = "-"

        rows.append(row)
        logger.info(
            "%s: n_seeds=%d, RMSE=%.4f+/-%.4f, signifikan=%s",
            prefix, row["n_seeds"], row.get("rmse_mean", float("nan")),
            row.get("rmse_std", float("nan")) or 0.0, row.get("n_significant_seeds", "-"),
        )

    if not rows:
        raise SystemExit("Tidak ada model dengan hasil yang bisa dimuat.")

    final_df = pd.DataFrame(rows)
    id_cols = ["model_prefix", "model_name", "domain", "n_seeds"]
    sig_cols = ["p_value_combined_fisher", "n_significant_seeds"]
    metric_cols = [c for c in final_df.columns if c not in id_cols + sig_cols]
    ordered_cols = [c for c in id_cols if c in final_df.columns] + metric_cols + sig_cols
    final_df = final_df[ordered_cols]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(final_df)
    logger.info("Tabel ringkasan manuskrip (%d model) disimpan ke %s", len(final_df), args.output)


if __name__ == "__main__":
    main()
