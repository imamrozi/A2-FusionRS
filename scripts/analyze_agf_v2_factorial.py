"""
scripts/analyze_agf_v2_factorial.py

Analisis konfirmasi TEST untuk arsitektur v2 -- memisahkan efek EKSTRAKSI,
SEQUENCE, dan TOKEN GLOBAL, dengan pembanding yang ADIL.

  | Sel | Ekstraksi        | Scorer  | Fusi | Token global | Sequence |
  |-----|------------------|---------|------|--------------|----------|
  | A   | leksikon keyword | SA-BERT | tree | implisit     | -        |
  | B'  | leksikon keyword | SA-BERT | AGF  | YA           | -        |
  | E   | PyABSA           | SA-BERT | AGF  | YA           | -        |
  | F   | PyABSA           | SA-BERT | AGF  | YA           | YA       |
  | F-  | PyABSA           | SA-BERT | AGF  | TIDAK        | YA       |

EFEK (RMSE lebih rendah = lebih baik; nilai NEGATIF = perbaikan):
  efek EKSTRAKSI PyABSA (ADIL)  : E  - B'
  efek representasi sequence    : F  - E
  efek token sentimen global    : F  - F-
  total vs A2-IRM               : F  - A

KLAIM YANG SAH vs KLAIM YANG MENYESATKAN
Kontribusi PyABSA HARUS dilaporkan sebagai `E - B'`, bukan `F - A`.
Alasannya (reports/gates_1_3_summary.md Bagian 4): token sentimen global
menolong KEDUA cabang -- ia fitur yang hilang dari representasi gaya
A2-IRM, bukan keunggulan PyABSA. Pada probe linier, selisih thd pembanding
naif kira-kira DUA KALI selisih thd pembanding adil. `F - A` tetap
dilaporkan sebagai konteks "total perbaikan sistem", TAPI harus dinyatakan
mencakup beberapa perubahan sekaligus.

Signifikansi: Wilcoxon berpasangan per-seed atas squared_error per-sampel,
+ Fisher-combined lintas seed -- konsisten dgn protokol A2-IRM.

Usage:
    python scripts/analyze_agf_v2_factorial.py
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

TEST_DIR = _REPO_ROOT / "checkpoints" / "results_phase2_clean_v2" / "test"
A2IRM_DIR = _REPO_ROOT / "checkpoints" / "results"
A2IRM_PREFIX = "absa_ablation_concat_confidence_cbf_nosentiment"

DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant", "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456, 789, 1011]

CELLS = {
    "A":  (A2IRM_DIR, A2IRM_PREFIX),
    "B'": (TEST_DIR, "agf_agf_keyword_cellBfair"),
    "E":  (TEST_DIR, "agf_a2fusionrs_clean_cellE"),
    "F":  (TEST_DIR, "agf_a2fusionrs_clean_cellF"),
    "F-": (TEST_DIR, "agf_a2fusionrs_clean_cellFminus"),
}
CELL_LABELS = {
    "A":  "A : keyword + tree (A2-IRM)",
    "B'": "B': keyword + AGF + global  [PEMBANDING ADIL]",
    "E":  "E : PyABSA + SA-BERT + AGF + global",
    "F":  "F : + sequence identitas aspek  [TARGET]",
    "F-": "F-: F tanpa token global (ablasi)",
}
EFFECTS = [
    ("efek EKSTRAKSI PyABSA (ADIL)", "E",  "B'"),
    ("efek representasi sequence",   "F",  "E"),
    ("efek token sentimen global",   "F",  "F-"),
    ("TOTAL vs A2-IRM (multi-faktor)", "F", "A"),
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
        if cell != "A" and data.get("eval_split") != "test":
            raise SystemExit(
                f"BERHENTI: {path.name} punya eval_split='{data.get('eval_split')}', "
                "bukan 'test'. Tabel konfirmasi harus memakai --stage confirm."
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
        da = pd.read_csv(pa)[["review_id", "squared_error"]].rename(
            columns={"squared_error": "se_a"})
        db = pd.read_csv(pb)[["review_id", "squared_error"]].rename(
            columns={"squared_error": "se_b"})
        m = da.merge(db, on="review_id", how="inner")
        if m.empty:
            continue
        _, p = significance_test(m["se_a"].values, m["se_b"].values, test="wilcoxon")
        pvals.append(p)
        n_sig += int(p < 0.05)
    if not pvals:
        return {"n_significant_seeds": "0/0", "p_fisher": None}
    _, combined = combine_pvalues(pvals, method="fisher")
    return {"n_significant_seeds": f"{n_sig}/{len(pvals)}", "p_fisher": combined}


def main() -> None:
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
    print("=" * 82)
    print("RMSE per sel (mean lintas 5 seed, TEST SET)")
    print("=" * 82)
    if not summary.empty:
        print(summary.pivot_table(index=["cell", "cell_label"], columns="domain",
                                  values="rmse_mean").to_string())

    if missing:
        print(f"\n!! Sel belum lengkap ({len(missing)}):")
        for m in missing:
            print("   -", m)
        raise SystemExit(
            "\nEfek TIDAK dihitung sampai semua sel lengkap. "
            "Jalankan scripts/run_agf_v2_factorial.sh (resumable)."
        )

    eff_rows = []
    for label, a, b in EFFECTS:
        for domain in DOMAINS:
            ra, rb = np.mean(load_rmse(a, domain)), np.mean(load_rmse(b, domain))
            eff_rows.append({
                "effect": label, "domain": DOMAIN_LABELS[domain], "cells": f"{a} - {b}",
                "rmse_a": ra, "rmse_b": rb, "delta": ra - rb,
                "pct_change": (ra - rb) / rb * 100,
                **paired_significance(a, b, domain),
            })
    effects = pd.DataFrame(eff_rows)
    print("\n\n" + "=" * 82)
    print("ESTIMASI EFEK (delta < 0 = PERBAIKAN)")
    print("=" * 82)
    print(effects.to_string(index=False))

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TEST_DIR / "v2_factorial_summary.csv", index=False)
    effects.to_csv(TEST_DIR / "v2_factorial_effects.csv", index=False)

    print("\n" + "=" * 82)
    print("CARA MELAPORKAN (wajib dipatuhi di manuskrip)")
    print("=" * 82)
    print(
        "1. Kontribusi ekstraksi PyABSA = 'E - B\\'', BUKAN 'F - A'. Token\n"
        "   sentimen global menolong KEDUA cabang, jadi ia fitur yang hilang\n"
        "   dari representasi gaya A2-IRM, bukan keunggulan PyABSA. Memakai\n"
        "   'F - A' sebagai kontribusi PyABSA melebih-lebihkan (~2x pada probe\n"
        "   linier) dan tidak akan bertahan di review.\n"
        "2. 'F - A' boleh dilaporkan sebagai TOTAL perbaikan sistem, dgn\n"
        "   pernyataan eksplisit bahwa ia mencakup beberapa perubahan.\n"
        "3. Efek token global ('F - F-') harus dilaporkan terpisah supaya\n"
        "   pembaca bisa menilai sendiri asal-usul perbaikannya."
    )
    print(f"\nDisimpan ke {TEST_DIR}/v2_factorial_{{summary,effects}}.csv")


if __name__ == "__main__":
    main()
