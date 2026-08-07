"""
scripts/analyze_agf_factorial.py

Tahap 7 (plan pure-painting-wilkes.md): analisis DESAIN FAKTORIAL 2x2 di
TEST SET -- memisahkan pengaruh penggantian ABSA (keyword -> PyABSA) dari
pengaruh penggantian FUSI (statis NMF+DT -> dinamis AGF).

  | Sel | Sumber ABSA         | Fusi          | Sumber hasil                  |
  |-----|---------------------|---------------|-------------------------------|
  |  A  | keyword concat+conf | statis NMF+DT | A2-IRM (checkpoints/results/) |
  |  B  | keyword concat+conf | dinamis AGF   | results_phase2_clean/test/    |
  |  C  | PyABSA rich 9-dim   | statis NMF+DT | idem                          |
  |  D0 | PyABSA rich 9-dim   | dinamis AGF   | idem                          |
  |  D  | PyABSA seq + rich   | dinamis AGF   | idem (ARSITEKTUR TARGET)      |

EFEK YANG DIESTIMASI (RMSE lebih rendah = lebih baik; nilai NEGATIF =
perbaikan):
  efek PyABSA pada fusi statis   : C  - A
  efek PyABSA pada fusi AGF      : D0 - B
  efek fusi AGF pada keyword     : B  - A
  efek fusi AGF pada PyABSA      : D0 - C
  interaksi ABSA x fusi          : (D0 - C) - (B - A)
  efek representasi sequence     : D  - D0
  total arsitektur bersih vs IRM : D  - A   (mencakup TIGA perubahan)

Signifikansi: Wilcoxon berpasangan per-seed atas squared_error per-sampel,
+ Fisher-combined lintas seed -- konsisten dgn protokol A2-IRM.

Usage:
    python scripts/analyze_agf_factorial.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import combine_pvalues

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import significance_test  # noqa: E402

TEST_DIR = _REPO_ROOT / "checkpoints" / "results_phase2_clean" / "test"
A2IRM_DIR = _REPO_ROOT / "checkpoints" / "results"
A2IRM_PREFIX = "absa_ablation_concat_confidence_cbf_nosentiment"

DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant", "amazon_electronics": "E-commerce", "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456, 789, 1011]

# sel -> (direktori, prefix file)
CELLS = {
    "A":  (A2IRM_DIR, A2IRM_PREFIX),
    "B":  (TEST_DIR, "agf_agf_keyword_cellB"),
    "C":  (TEST_DIR, "agf_static_pyabsa_rich_cellC"),
    "D0": (TEST_DIR, "agf_a2fusionrs_clean_cellD0"),
    "D":  (TEST_DIR, "agf_a2fusionrs_clean_cellD"),
}
CELL_LABELS = {
    "A": "A: keyword + tree (A2-IRM)",
    "B": "B: keyword + AGF",
    "C": "C: PyABSA-rich + tree",
    "D0": "D0: PyABSA-rich + AGF",
    "D": "D: PyABSA seq+rich + AGF",
}
# (label efek, sel_a, sel_b) -> dilaporkan sbg a - b
EFFECTS = [
    ("efek PyABSA (fusi statis)",      "C",  "A"),
    ("efek PyABSA (fusi AGF)",         "D0", "B"),
    ("efek fusi AGF (ABSA keyword)",   "B",  "A"),
    ("efek fusi AGF (ABSA PyABSA)",    "D0", "C"),
    ("efek representasi sequence",     "D",  "D0"),
    ("TOTAL bersih vs A2-IRM",         "D",  "A"),
]


def load_rmse(cell: str, domain: str) -> list[float]:
    d, prefix = CELLS[cell]
    vals = []
    for seed in SEEDS:
        path = d / f"{prefix}_{domain}_seed{seed}.yaml"
        if not path.exists():
            continue
        with open(path) as f:
            data = yaml.safe_load(f)
        # Pengaman: sel B/C/D0/D HARUS berasal dari stage=confirm (test set).
        if cell != "A" and data.get("eval_split") != "test":
            raise SystemExit(
                f"BERHENTI: {path.name} punya eval_split='{data.get('eval_split')}', "
                "bukan 'test'. Tabel faktorial harus memakai hasil --stage confirm."
            )
        vals.append(data["rmse"])
    return vals


def paired_significance(cell_a: str, cell_b: str, domain: str) -> dict:
    dir_a, prefix_a = CELLS[cell_a]
    dir_b, prefix_b = CELLS[cell_b]
    pvals, n_sig = [], 0
    for seed in SEEDS:
        pa = dir_a / f"predictions_{prefix_a}_{domain}_seed{seed}.csv"
        pb = dir_b / f"predictions_{prefix_b}_{domain}_seed{seed}.csv"
        if not pa.exists() or not pb.exists():
            continue
        da = pd.read_csv(pa)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_a"})
        db = pd.read_csv(pb)[["review_id", "squared_error"]].rename(columns={"squared_error": "se_b"})
        m = da.merge(db, on="review_id", how="inner")
        if m.empty:
            continue
        _, p = significance_test(m["se_a"].values, m["se_b"].values, test="wilcoxon")
        pvals.append(p)
        if p < 0.05:
            n_sig += 1
    if not pvals:
        return {"n_significant_seeds": "0/0", "p_fisher": None}
    _, combined = combine_pvalues(pvals, method="fisher")
    return {"n_significant_seeds": f"{n_sig}/{len(pvals)}", "p_fisher": combined}


def main() -> None:
    # ---- Tabel deskriptif ----
    rows, missing = [], []
    for cell in CELLS:
        for domain in DOMAINS:
            vals = load_rmse(cell, domain)
            if len(vals) < len(SEEDS):
                missing.append(f"{cell}/{domain}: {len(vals)}/{len(SEEDS)} seed")
            if not vals:
                continue
            rows.append({
                "cell": cell, "cell_label": CELL_LABELS[cell],
                "domain": DOMAIN_LABELS[domain], "n_seeds": len(vals),
                "rmse_mean": float(np.mean(vals)),
                "rmse_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=== RMSE per sel faktorial (mean +/- SD lintas seed, TEST SET) ===")
    if not summary.empty:
        print(summary.pivot_table(index=["cell", "cell_label"], columns="domain",
                                  values="rmse_mean").to_string())

    if missing:
        print(f"\n!! Sel belum lengkap ({len(missing)}):")
        for m in missing:
            print("   -", m)
        print("\nEfek TIDAK dihitung sampai semua sel lengkap.")
        raise SystemExit(
            "Jalankan scripts/run_agf_clean_factorial.sh (resumable) sampai 60/60."
        )

    # ---- Estimasi efek + signifikansi ----
    eff_rows = []
    for label, a, b in EFFECTS:
        for domain in DOMAINS:
            ra, rb = np.mean(load_rmse(a, domain)), np.mean(load_rmse(b, domain))
            sig = paired_significance(a, b, domain)
            eff_rows.append({
                "effect": label, "domain": DOMAIN_LABELS[domain],
                "cells": f"{a} - {b}",
                "rmse_a": ra, "rmse_b": rb,
                "delta": ra - rb, "pct_change": (ra - rb) / rb * 100,
                **sig,
            })
    effects = pd.DataFrame(eff_rows)
    print("\n\n=== ESTIMASI EFEK (delta<0 = PERBAIKAN) ===")
    print(effects.to_string(index=False))

    # ---- Interaksi ----
    print("\n\n=== INTERAKSI ABSA x FUSI: (D0-C) - (B-A) ===")
    print("(positif = efek fusi AGF LEBIH KECIL pada PyABSA drpd pada keyword)")
    for domain in DOMAINS:
        e_fusi_pyabsa = np.mean(load_rmse("D0", domain)) - np.mean(load_rmse("C", domain))
        e_fusi_keyword = np.mean(load_rmse("B", domain)) - np.mean(load_rmse("A", domain))
        print(f"  {DOMAIN_LABELS[domain]:12}: {e_fusi_pyabsa - e_fusi_keyword:+.4f} "
              f"(fusi|PyABSA={e_fusi_pyabsa:+.4f}, fusi|keyword={e_fusi_keyword:+.4f})")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TEST_DIR / "factorial_summary.csv", index=False)
    effects.to_csv(TEST_DIR / "factorial_effects.csv", index=False)
    print(f"\nDisimpan ke {TEST_DIR}/factorial_summary.csv & factorial_effects.csv")
    print(
        "\nCATATAN WAJIB utk manuskrip: sel C & D0 memakai PyABSA rich 9-dim (agregasi),\n"
        "BUKAN sequence, karena tree NMF+DT secara struktural tidak bisa mengkonsumsi\n"
        "sequence panjang-variabel. Karena itu 'TOTAL bersih vs A2-IRM' (D-A) mencakup\n"
        "TIGA perubahan sekaligus; dekomposisinya ada di baris-baris efek di atas."
    )


if __name__ == "__main__":
    main()
