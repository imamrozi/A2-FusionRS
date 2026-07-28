"""
export_manuscript_data.py

Ekspor SATU sumber kebenaran (source of truth) dari seluruh hasil
eksperimen A2-IRM (rerun 90-run AdamW + floor no_sentiment_ablation yang
sudah diperbaiki) ke CSV (long-format, per seed) dan XLSX (multi-sheet,
termasuk tabel ringkasan siap-pakai utk manuskrip).

Dipakai utk finalisasi manuskrip di luar sesi ini (mis. Claude Project) --
supaya tidak perlu re-derive angka dari checkpoints/results/*.yaml mentah.

Usage:
    python scripts/export_manuscript_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import combine_pvalues

from src.evaluation.metrics import significance_test

RESULTS_DIR = Path("checkpoints/results")
ARCHIVE_DIR = Path("checkpoints/results_pre_adamw_fix_2026-07-26")
OUT_DIR = Path("manuscript/data_export")
SEEDS = [42, 123, 456, 789, 1011]
DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {"restaurant": "Restaurant", "amazon_electronics": "E-commerce", "tripadvisor_hotel": "Hotel"}

# (prefix, display name, is_floor)
MODEL_VARIANTS = [
    ("baseline_reimpl_cbf_nosentiment", "Baseline (global SA)"),
    ("absa_ablation_cbf_nosentiment", "ABSA mean"),
    ("absa_ablation_confidence_mean_cbf_nosentiment", "ABSA confidence-mean"),
    ("absa_ablation_concat_cbf_nosentiment", "ABSA concat"),
    ("absa_ablation_concat_confidence_cbf_nosentiment", "ABSA concat+confidence"),
]
# Floor prefix changed after the NMF-degeneracy fix (2026-07-28); "_fixed_v2"
# results are the trustworthy ones -- old (buggy, constant-column) floor
# results are kept in raw_per_seed for transparency but flagged invalid.
FLOOR_PREFIX_FIXED = "absa_ablation_concat_confidence_cbf_nosentiment_no_sentiment_fixed_v2"
FLOOR_PREFIX_BUGGY = "absa_ablation_concat_confidence_cbf_nosentiment_no_sentiment"


def load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def build_raw_per_seed() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for prefix, label in MODEL_VARIANTS:
            for seed in SEEDS:
                d = load_yaml(RESULTS_DIR / f"{prefix}_{domain}_seed{seed}.yaml")
                if d is None:
                    continue
                rows.append({
                    "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                    "model_prefix": prefix, "model_label": label,
                    "seed": seed, "rmse": d["rmse"], "mae": d["mae"],
                    "n_test_samples": d.get("n_test_samples"),
                    "is_floor": False, "floor_status": "",
                })
        # Floor -- buggy (pre-fix) version, kept for transparency
        for seed in SEEDS:
            d = load_yaml(RESULTS_DIR / f"{FLOOR_PREFIX_BUGGY}_{domain}_seed{seed}.yaml")
            if d is not None:
                rows.append({
                    "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                    "model_prefix": FLOOR_PREFIX_BUGGY, "model_label": "No-sentiment floor (BUGGY, do not use)",
                    "seed": seed, "rmse": d["rmse"], "mae": d["mae"],
                    "n_test_samples": d.get("n_test_samples"),
                    "is_floor": True, "floor_status": "buggy_constant_column_nmf",
                })
        # Floor -- fixed version
        for seed in SEEDS:
            d = load_yaml(RESULTS_DIR / f"{FLOOR_PREFIX_FIXED}_{domain}_seed{seed}.yaml")
            if d is not None:
                rows.append({
                    "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                    "model_prefix": FLOOR_PREFIX_FIXED, "model_label": "No-sentiment floor (fixed)",
                    "seed": seed, "rmse": d["rmse"], "mae": d["mae"],
                    "n_test_samples": d.get("n_test_samples"),
                    "is_floor": True, "floor_status": "fixed_dropped_column",
                })
        # Classical CF (archived -- unaffected by DeepMF AdamW fix)
        for prefix, label in [("classical_cf_item_knn", "Item-KNN"), ("classical_cf_svd", "SVD")]:
            for seed in SEEDS:
                d = load_yaml(ARCHIVE_DIR / f"{prefix}_{domain}_seed{seed}.yaml")
                if d is not None:
                    rows.append({
                        "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                        "model_prefix": prefix, "model_label": label,
                        "seed": seed, "rmse": d["rmse"], "mae": d["mae"],
                        "n_test_samples": d.get("n_test_samples"),
                        "is_floor": False, "floor_status": "",
                    })
    return pd.DataFrame(rows)


def paired_significance(prefix_a: str, prefix_b: str, domain: str, results_dir_a: Path, results_dir_b: Path) -> tuple[str, float | None]:
    n_sig, pvals = 0, []
    for seed in SEEDS:
        path_a = results_dir_a / f"predictions_{prefix_a}_{domain}_seed{seed}.csv"
        path_b = results_dir_b / f"predictions_{prefix_b}_{domain}_seed{seed}.csv"
        if not path_a.exists() or not path_b.exists():
            continue
        df_a = pd.read_csv(path_a)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_a"})
        df_b = pd.read_csv(path_b)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_b"})
        merged = df_a.merge(df_b, on="review_id", how="inner")
        if merged.empty:
            continue
        _, p = significance_test(merged["se_a"].values, merged["se_b"].values, test="wilcoxon")
        pvals.append(p)
        if p < 0.05:
            n_sig += 1
    if not pvals:
        return "0/0", None
    _, combined = combine_pvalues(pvals, method="fisher")
    return f"{n_sig}/{len(pvals)}", combined


def build_summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        sub = raw[(raw["domain"] == domain) & (~raw["is_floor"])]
        for prefix, label in MODEL_VARIANTS + [("classical_cf_item_knn", "Item-KNN"), ("classical_cf_svd", "SVD")]:
            group = sub[sub["model_prefix"] == prefix]
            if group.empty:
                continue
            is_baseline = prefix == "baseline_reimpl_cbf_nosentiment"
            results_dir = ARCHIVE_DIR if prefix.startswith("classical_cf") else RESULTS_DIR
            if is_baseline:
                n_sig, p_combined = "-", None
            else:
                n_sig, p_combined = paired_significance(
                    "baseline_reimpl_cbf_nosentiment", prefix, domain, RESULTS_DIR, results_dir,
                )
            rows.append({
                "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                "model_prefix": prefix, "model_label": label,
                "n_seeds": len(group),
                "rmse_mean": group["rmse"].mean(), "rmse_std": group["rmse"].std(ddof=1),
                "mae_mean": group["mae"].mean(), "mae_std": group["mae"].std(ddof=1),
                "rmse_pct_change_vs_baseline": None,
                "n_significant_seeds": n_sig, "p_value_fisher_combined": p_combined,
            })
        # Floor (fixed only) -- comparable RMSE/MAE, no paired significance vs baseline
        # computed here (different feature-set model, comparison is descriptive).
        floor_group = raw[(raw["domain"] == domain) & (raw["model_prefix"] == FLOOR_PREFIX_FIXED)]
        if not floor_group.empty:
            rows.append({
                "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                "model_prefix": FLOOR_PREFIX_FIXED, "model_label": "No-sentiment floor (fixed)",
                "n_seeds": len(floor_group),
                "rmse_mean": floor_group["rmse"].mean(), "rmse_std": floor_group["rmse"].std(ddof=1),
                "mae_mean": floor_group["mae"].mean(), "mae_std": floor_group["mae"].std(ddof=1),
                "rmse_pct_change_vs_baseline": None,
                "n_significant_seeds": "n/a", "p_value_fisher_combined": None,
            })

    df = pd.DataFrame(rows)
    baseline_rmse = df[df["model_prefix"] == "baseline_reimpl_cbf_nosentiment"].set_index("domain")["rmse_mean"]
    df["rmse_pct_change_vs_baseline"] = df.apply(
        lambda r: (r["rmse_mean"] - baseline_rmse[r["domain"]]) / baseline_rmse[r["domain"]] * 100
        if r["domain"] in baseline_rmse.index else None,
        axis=1,
    )
    return df


def build_dataset_characteristics() -> pd.DataFrame:
    return pd.DataFrame([
        {"domain": "restaurant", "domain_label": "Restaurant (Yelp)", "reviews": 118695, "users": 7152, "items": 3757, "sparsity_pct": 99.56, "mean_rating": 3.76, "test_size": 13233},
        {"domain": "amazon_electronics", "domain_label": "E-commerce (Amazon Electronics)", "reviews": 122068, "users": 14750, "items": 9226, "sparsity_pct": 99.91, "mean_rating": 4.37, "test_size": 16580},
        {"domain": "tripadvisor_hotel", "domain_label": "Hotel (TripAdvisor)", "reviews": 79562, "users": 11236, "items": 2056, "sparsity_pct": 99.66, "mean_rating": 3.94, "test_size": 11795},
    ])


def build_aspect_coverage() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        d = load_yaml(RESULTS_DIR / f"absa_ablation_concat_cbf_nosentiment_{domain}_seed42.yaml")
        cov = d.get("aspect_coverage", {}) if d else {}
        rows.append({
            "domain": domain, "domain_label": DOMAIN_LABELS[domain],
            "n_reviews": cov.get("n_reviews"),
            "pct_with_any_aspect_match": round(cov.get("pct_with_any_aspect_match", 0) * 100, 2),
            "aspect_match_pct_detail": str(cov.get("aspect_match_pct", {})),
        })
    return pd.DataFrame(rows)


def build_table4_variance() -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        for prefix, label in [
            ("baseline_reimpl_cbf_nosentiment", "Baseline"),
            ("absa_ablation_concat_confidence_cbf_nosentiment", "Concat+confidence"),
        ]:
            vals, seed_map = [], {}
            for seed in SEEDS:
                d = load_yaml(RESULTS_DIR / f"{prefix}_{domain}_seed{seed}.yaml")
                if d is None:
                    continue
                vals.append(d["rmse"])
                seed_map[len(vals) - 1] = seed
            arr = np.array(vals)
            if len(arr) < 2:
                continue
            full_sd = arr.std(ddof=1)
            z = np.abs((arr - arr.mean()) / full_sd) if full_sd > 0 else np.zeros_like(arr)
            idx = int(np.argmax(z))
            arr_excl = np.delete(arr, idx)
            excl_sd = arr_excl.std(ddof=1)
            rows.append({
                "domain": domain, "domain_label": DOMAIN_LABELS[domain],
                "model_label": label, "full_sd": full_sd, "sd_excl_extreme_seed": excl_sd,
                "internal_reduction_factor": full_sd / excl_sd if excl_sd > 0 else None,
                "extreme_seed": seed_map[idx],
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = build_raw_per_seed()
    summary = build_summary_table(raw)
    dataset_chars = build_dataset_characteristics()
    coverage = build_aspect_coverage()
    table4 = build_table4_variance()

    raw.to_csv(OUT_DIR / "raw_per_seed_results.csv", index=False)
    summary.to_csv(OUT_DIR / "table3_summary_with_significance.csv", index=False)
    dataset_chars.to_csv(OUT_DIR / "table1_dataset_characteristics.csv", index=False)
    coverage.to_csv(OUT_DIR / "aspect_coverage.csv", index=False)
    table4.to_csv(OUT_DIR / "table4_variance_analysis.csv", index=False)

    xlsx_path = OUT_DIR / "A2-IRM_results_source_of_truth.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        dataset_chars.to_excel(writer, sheet_name="Table1_Dataset", index=False)
        summary.to_excel(writer, sheet_name="Table3_Summary_Significance", index=False)
        table4.to_excel(writer, sheet_name="Table4_Variance_Analysis", index=False)
        coverage.to_excel(writer, sheet_name="Aspect_Coverage", index=False)
        raw.to_excel(writer, sheet_name="Raw_Per_Seed", index=False)

        # Auto-width columns (readability) -- openpyxl doesn't do this by default.
        for sheet in writer.sheets.values():
            for col_cells in sheet.columns:
                length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                col_letter = col_cells[0].column_letter
                sheet.column_dimensions[col_letter].width = min(max(length + 2, 10), 60)

    print(f"CSV files -> {OUT_DIR}/")
    print(f"XLSX      -> {xlsx_path}")
    print(f"raw_per_seed rows: {len(raw)}")
    print(f"summary rows: {len(summary)}")
    n_floor_fixed = (raw["model_prefix"] == FLOOR_PREFIX_FIXED).sum()
    n_floor_expected = len(DOMAINS) * len(SEEDS)
    print(f"floor (fixed) rows present: {n_floor_fixed}/{n_floor_expected}" +
          (" -- INCOMPLETE, some domains still running" if n_floor_fixed < n_floor_expected else " -- complete"))


if __name__ == "__main__":
    main()
