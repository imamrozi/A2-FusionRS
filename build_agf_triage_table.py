"""
build_agf_triage_table.py

Stage G (plan pure-painting-wilkes.md): analisis 45-run triage AGF
(scripts/rerun_agf_triage.sh) terhadap A2-IRM yang SUDAH DIPERBAIKI
(concat_confidence, checkpoints/results/absa_ablation_concat_confidence_
cbf_nosentiment_*.yaml, dari matriks 90-run AdamW).

4 uji signifikansi berpasangan (Wilcoxon per-seed + Fisher-combined),
menjawab LANGSUNG kedua klaim verdict lama (memori phase2-agf-final-
verdict) di atas baseline yang benar:

A. agf_keyword_oof_perseq vs A2-IRM (concat_confidence)
   -- KLAIM HEADLINE: apakah A2-FusionRS >> A2-IRM masih berlaku?
B. agf_keyword_oof_perseq vs static_pyabsa (kontrol atribusi TIDAK ADIL --
   lihat catatan di bawah, dipertahankan hanya utk transparansi/riwayat)
C. static_pyabsa vs A2-IRM (concat_confidence)
   -- pelengkap B: apakah PyABSA SENDIRI (tanpa AGF sama sekali) sudah
      cukup mengalahkan A2-IRM?
D. agf_keyword_oof_perseq vs agf_keyword (polos, tanpa redesign)
   -- apakah redesign (asymmetric + residual OOF + perseq) memberi nilai
      tambah nyata dibanding AGF+keyword paling dasar?
E. agf_keyword_oof_perseq vs static_keyword_pyabsa (KONTROL ATRIBUSI YANG
   ADIL -- info sama persis: keyword ABSA concat+confidence HSTACK PyABSA-
   rich, cuma beda mekanisme fusi: AGF adaptif vs tree statis)
   -- INI klaim atribusi verdict lama yang SEBENARNYA diuji ulang: kalau E
      tidak signifikan/selisih kecil -> PyABSA/info tambahan yang
      komplementer (bukan attention). Kalau E signifikan & E jauh lebih
      kecil dari B (selisih AGF vs static_pyabsa) -> sebagian AGF vs
      static_pyabsa lama (B) memang cuma "AGF dapat info lebih banyak",
      TAPI attention/gating sendiri juga menambah nilai di atas info yang
      sama.

CATATAN METODOLOGIS (ditambahkan setelah triage 45-run awal): perbandingan
B (agf_keyword_oof_perseq vs static_pyabsa) TIDAK adil -- static_pyabsa
cuma menerima PyABSA 5-dim summary (TANPA keyword ABSA), sedangkan
agf_keyword_oof_perseq menerima keyword ABSA + PyABSA aspect-sequence +
base OOF sekaligus. Selisih besar di B sebagian besar mencerminkan "info
lebih banyak", bukan murni "mekanisme fusi lebih baik". Perbandingan E
(ditambah scripts/rerun_agf_attribution_control.sh, 15 run lanjutan)
menyamakan info kedua sisi -- INI yang jadi rujukan utama utk menjawab
klaim atribusi verdict lama, bukan B.

Usage:
    python build_agf_triage_table.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import combine_pvalues

from src.evaluation.metrics import significance_test

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("checkpoints/results")
DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {"restaurant": "Restaurant", "amazon_electronics": "E-commerce", "tripadvisor_hotel": "Hotel"}
SEEDS = [42, 123, 456, 789, 1011]

A2IRM_REFERENCE_PREFIX = "absa_ablation_concat_confidence_cbf_nosentiment"

# (label singkat, prefix file hasil, direktori)
MODELS = [
    ("A2-IRM (concat_confidence)", A2IRM_REFERENCE_PREFIX, RESULTS_DIR),
    ("static_pyabsa", "agf_static_pyabsa", RESULTS_DIR),
    ("static_keyword_pyabsa (kontrol atribusi adil)", "agf_static_keyword_pyabsa", RESULTS_DIR),
    ("agf_keyword (polos)", "agf_agf_keyword", RESULTS_DIR),
    ("agf_keyword_oof_perseq (A2-FusionRS penuh)", "agf_agf_keyword_oof_perseq", RESULTS_DIR),
]

# (label, prefix_a, prefix_b) -- 5 uji signifikansi A-E, lihat docstring modul
COMPARISONS = [
    ("headline_vs_A2IRM", "agf_agf_keyword_oof_perseq", A2IRM_REFERENCE_PREFIX),
    ("B_atribusi_TIDAK_ADIL_vs_static_pyabsa", "agf_agf_keyword_oof_perseq", "agf_static_pyabsa"),
    ("static_pyabsa_vs_A2IRM", "agf_static_pyabsa", A2IRM_REFERENCE_PREFIX),
    ("redesign_vs_agf_keyword_polos", "agf_agf_keyword_oof_perseq", "agf_agf_keyword"),
    ("E_atribusi_ADIL_vs_static_keyword_pyabsa", "agf_agf_keyword_oof_perseq", "agf_static_keyword_pyabsa"),
    ("static_keyword_pyabsa_vs_A2IRM", "agf_static_keyword_pyabsa", A2IRM_REFERENCE_PREFIX),
]


def load_rmse(prefix: str, domain: str) -> list[float]:
    vals = []
    for seed in SEEDS:
        path = RESULTS_DIR / f"{prefix}_{domain}_seed{seed}.yaml"
        if not path.exists():
            logger.warning("Tidak ditemukan: %s", path)
            continue
        with open(path) as f:
            d = yaml.safe_load(f)
        vals.append(d["rmse"])
    return vals


def paired_significance(prefix_a: str, prefix_b: str, domain: str) -> dict:
    p_values = []
    n_significant = 0
    for seed in SEEDS:
        path_a = RESULTS_DIR / f"predictions_{prefix_a}_{domain}_seed{seed}.csv"
        path_b = RESULTS_DIR / f"predictions_{prefix_b}_{domain}_seed{seed}.csv"
        if not path_a.exists() or not path_b.exists():
            continue
        df_a = pd.read_csv(path_a)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_a"})
        df_b = pd.read_csv(path_b)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_b"})
        merged = df_a.merge(df_b, on="review_id", how="inner")
        if merged.empty:
            continue
        _, p = significance_test(merged["se_a"].values, merged["se_b"].values, test="wilcoxon")
        p_values.append(p)
        if p < 0.05:
            n_significant += 1
    if not p_values:
        return {"n_significant_seeds": "0/0", "p_value_fisher": None}
    _, combined_p = combine_pvalues(p_values, method="fisher")
    return {"n_significant_seeds": f"{n_significant}/{len(p_values)}", "p_value_fisher": combined_p}


def main() -> None:
    # ---- Tabel deskriptif: mean +/- SD RMSE per model x domain ----
    summary_rows = []
    for label, prefix, _ in MODELS:
        for domain in DOMAINS:
            vals = load_rmse(prefix, domain)
            if not vals:
                continue
            summary_rows.append({
                "model": label, "domain": DOMAIN_LABELS[domain], "n_seeds": len(vals),
                "rmse_mean": float(np.mean(vals)), "rmse_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "agf_triage_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---- 4 uji signifikansi x 3 domain ----
    sig_rows = []
    for comp_label, prefix_a, prefix_b in COMPARISONS:
        for domain in DOMAINS:
            rmse_a = load_rmse(prefix_a, domain)
            rmse_b = load_rmse(prefix_b, domain)
            result = paired_significance(prefix_a, prefix_b, domain)
            rmse_a_mean = float(np.mean(rmse_a)) if rmse_a else None
            rmse_b_mean = float(np.mean(rmse_b)) if rmse_b else None
            pct_change = (
                (rmse_a_mean - rmse_b_mean) / rmse_b_mean * 100
                if rmse_a_mean is not None and rmse_b_mean is not None else None
            )
            sig_rows.append({
                "comparison": comp_label, "domain": DOMAIN_LABELS[domain],
                "model_a": prefix_a, "rmse_a": rmse_a_mean,
                "model_b": prefix_b, "rmse_b": rmse_b_mean,
                "pct_change_a_vs_b": pct_change,
                **result,
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_path = RESULTS_DIR / "agf_triage_significance.csv"
    sig_df.to_csv(sig_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== RINGKASAN RMSE (mean +/- SD, 5 seed) ===")
    print(summary_df.to_string(index=False))
    print("\n=== UJI SIGNIFIKANSI (Wilcoxon per-seed + Fisher-combined) ===")
    print(sig_df.to_string(index=False))
    logger.info("Disimpan ke %s dan %s", summary_path, sig_path)


if __name__ == "__main__":
    main()
