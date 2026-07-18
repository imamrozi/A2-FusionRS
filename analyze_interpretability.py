"""
analyze_interpretability.py

Analisis interpretability A2-FusionRS (§6.5 manuskrip), 3 eksperimen:

- Exp-A (bobot gate per-modalitas): dari `gates_agf_agf_keyword_oof_perseq_*.csv`
  (SUDAH tersimpan run multi-seed). Rata-rata bobot gate tiap modalitas per
  domain + uji korelasi gate modalitas PyABSA-aspek vs cakupan keyword-ABSA
  domain (hipotesis: makin tinggi cakupan aspek, makin besar peran modalitas
  aspek -> sinyal interpretability INDEPENDEN yg mengonfirmasi temuan akurasi
  ketergantungan-cakupan).
- Exp-B (studi kasus atensi aspek): dari `interp_cases_*.csv` (hasil
  --export-interpretability). Pilih contoh ilustratif (atensi terkonsentrasi,
  campuran aspek pos/neg) utk tabel kualitatif.
- Exp-C (faithfulness): dari `interp_faithfulness_*.csv`. Agregasi |Δ|top vs
  acak lintas domain.

CATATAN JUJUR (untuk teks manuskrip): bobot gate/atensi menjelaskan KOREKSI
AGF di atas base statis (struktur residual), bukan seluruh prediksi; dan
atensi = atribusi yg PLAUSIBEL, divalidasi faithfulness (Exp-C), bukan
penjelasan yg dijamin faithful (lih. debat Jain & Wallace 2019 vs Wiegreffe
& Pinter 2019).

Usage:
    python analyze_interpretability.py --results-dir checkpoints/results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["amazon_electronics", "restaurant", "tripadvisor_hotel"]
SEEDS = [42, 123, 456, 789, 1011]
KEYWORD_COVERAGE = {"amazon_electronics": 45.1, "restaurant": 87.7, "tripadvisor_hotel": 95.9}
PERSEQ_STEM = "agf_agf_keyword_oof_perseq"


def exp_a_gate_weights(results_dir: Path) -> pd.DataFrame:
    """Rata-rata bobot gate per modalitas per domain (lintas seed) + korelasi
    gate PyABSA-aspek vs cakupan."""
    rows = []
    for domain in DOMAINS:
        per_seed = []
        for seed in SEEDS:
            p = results_dir / f"gates_{PERSEQ_STEM}_{domain}_seed{seed}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            gate_cols = [c for c in df.columns if c.startswith("gate_")]
            per_seed.append(df[gate_cols].mean())
        if not per_seed:
            logger.warning("Tak ada gates_*.csv utk %s -- dilewati.", domain)
            continue
        mean_across = pd.concat(per_seed, axis=1).mean(axis=1)
        row = {"domain": domain, "coverage_keyword": KEYWORD_COVERAGE[domain], "n_seeds": len(per_seed)}
        row.update({c: round(float(mean_across[c]), 4) for c in mean_across.index})
        rows.append(row)
    table = pd.DataFrame(rows)
    logger.info("\n=== Exp-A: rata-rata bobot gate per modalitas per domain ===\n%s", table.to_string(index=False))

    if "gate_pyabsa_aspect" in table.columns and len(table) >= 2:
        r = np.corrcoef(table["coverage_keyword"], table["gate_pyabsa_aspect"])[0, 1]
        logger.info(
            "Korelasi Pearson (cakupan keyword vs bobot gate PyABSA-aspek): r=%.3f "
            "(%s hipotesis: aspek lebih berperan di domain cakupan-tinggi).",
            r, "MENDUKUNG" if r > 0 else "TIDAK mendukung",
        )
    return table


def exp_b_case_studies(results_dir: Path, domain: str, seed: int = 42, top_n: int = 8) -> None:
    """Pilih contoh ilustratif: atensi terkonsentrasi (satu aspek dominan) +
    ada aspek negatif kuat, prediksi menyimpang dari 5 (agar cerita jelas)."""
    p = results_dir / f"interp_cases_{PERSEQ_STEM}_{domain}_seed{seed}.csv"
    if not p.exists():
        logger.warning("Tak ada %s -- lewati Exp-B utk %s.", p.name, domain)
        return
    df = pd.read_csv(p)
    df = df[df["n_aspects"] >= 2].copy()

    def max_attn(s):
        return max(float(x) for x in str(s).split("|"))

    def min_ppos(s):
        return min(float(x) for x in str(s).split("|"))

    df["attn_max"] = df["attn"].map(max_attn)
    df["ppos_min"] = df["p_pos"].map(min_ppos)  # ada aspek yg sangat negatif?
    # skor keterbacaan: atensi terkonsentrasi & ada aspek negatif
    df["case_score"] = df["attn_max"] * (1.0 - df["ppos_min"])
    top = df.sort_values("case_score", ascending=False).head(top_n)
    logger.info(
        "\n=== Exp-B: %s (seed %d) -- %d studi kasus atensi aspek teratas ===\n%s",
        domain, seed, len(top),
        top[["review_id", "n_aspects", "pred", "actual", "aspects", "attn", "p_pos"]].to_string(index=False),
    )


def exp_c_faithfulness(results_dir: Path) -> pd.DataFrame:
    """Agregasi uji faithfulness (|Δ|top vs acak) lintas domain (seed 42)."""
    rows = []
    for domain in DOMAINS:
        p = results_dir / f"interp_faithfulness_{PERSEQ_STEM}_{domain}_seed42.csv"
        if not p.exists():
            logger.warning("Tak ada %s -- lewati.", p.name)
            continue
        d = pd.read_csv(p).iloc[0].to_dict()
        d["domain"] = domain
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)[
        ["domain", "n_rows_ge2_aspects", "mean_delta_top_attended", "mean_delta_random", "frac_top_gt_random", "wilcoxon_p"]
    ]
    logger.info(
        "\n=== Exp-C: faithfulness (buang aspek top-atensi vs acak) ===\n%s",
        table.to_string(index=False),
    )
    logger.info(
        "Interpretasi: |Δ|top > |Δ|acak & frac>0.5 => atensi mencerminkan pengaruh nyata "
        "(atribusi plausibel & faithful), bukan sekadar dekoratif."
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis interpretability A2-FusionRS (§6.5)")
    parser.add_argument("--results-dir", type=str, default="checkpoints/results")
    parser.add_argument("--seed", type=int, default=42, help="Seed utk studi kasus Exp-B.")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    exp_a_gate_weights(results_dir)
    for domain in DOMAINS:
        exp_b_case_studies(results_dir, domain, seed=args.seed)
    exp_c_faithfulness(results_dir)


if __name__ == "__main__":
    main()
