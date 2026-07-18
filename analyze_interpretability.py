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
        # r<0 = modalitas PyABSA-aspek dapat bobot LEBIH BESAR di domain
        # cakupan-RENDAH -> INDEPENDEN mengonfirmasi temuan akurasi (manfaat
        # PyABSA terbesar saat keyword-ABSA lemah). Bukan hukum monoton (n=3,
        # Hotel outlier utama) tapi arahnya konsisten -- triangulasi.
        arah = (
            "KONSISTEN dgn temuan akurasi (aspek lebih diandalkan saat cakupan keyword RENDAH)"
            if r < 0 else "berlawanan dgn temuan akurasi -- periksa"
        )
        logger.info(
            "Korelasi Pearson (cakupan keyword vs bobot gate PyABSA-aspek): r=%.3f -- %s. "
            "(n=3 domain: indikatif, bukan konklusif; Hotel = cakupan tertinggi & bobot aspek terendah.)",
            r, arah,
        )
    return table


def exp_b_case_studies(results_dir: Path, domain: str, seed: int = 42, top_n: int = 6) -> None:
    """Pilih contoh ILUSTRATIF yg BERSIH & JUJUR utk tabel manuskrip: aspek
    top-atensi harus BERNAMA (bukan <UNK>), atensi terkonsentrasi, dan sentimen
    aspek-top SEARAH prediksi (koheren). Ini ilustrasi tipikal -- bukti rigor
    tetap Exp-C (faithfulness agregat ~70%, bukan 100%; tak semua kasus koheren,
    itu dilaporkan jujur)."""
    p = results_dir / f"interp_cases_{PERSEQ_STEM}_{domain}_seed{seed}.csv"
    if not p.exists():
        logger.warning("Tak ada %s -- lewati Exp-B utk %s.", p.name, domain)
        return
    df = pd.read_csv(p)
    df = df[df["n_aspects"] >= 2].copy()

    records = []
    for _, row in df.iterrows():
        aspects = str(row["aspects"]).split("|")
        attn = [float(x) for x in str(row["attn"]).split("|")]
        ppos = [float(x) for x in str(row["p_pos"]).split("|")]
        if len(aspects) != len(attn) or len(aspects) != len(ppos):
            continue
        j = int(np.argmax(attn))
        top_aspect, top_attn, top_ppos = aspects[j], attn[j], ppos[j]
        if top_aspect in ("<UNK>", "<PAD>"):
            continue  # ilustrasi butuh aspek bernama
        pred = float(row["pred"])
        # koheren: aspek-top positif & pred tinggi, ATAU negatif & pred rendah
        coherent = (top_ppos > 0.6 and pred > 3.5) or (top_ppos < 0.4 and pred < 3.0)
        if not coherent:
            continue
        records.append({
            "review_id": row["review_id"], "n_aspects": int(row["n_aspects"]),
            "top_aspect": top_aspect, "top_attn": round(top_attn, 3),
            "top_sentiment": "POS" if top_ppos > 0.6 else "NEG", "top_p_pos": round(top_ppos, 2),
            "pred": round(pred, 2), "actual": row["actual"], "all_aspects": row["aspects"],
        })
    if not records:
        logger.warning("Exp-B %s: tak ada kasus koheren+bernama (semua top-atensi UNK/tak koheren).", domain)
        return
    out = pd.DataFrame(records).sort_values("top_attn", ascending=False).head(top_n)
    logger.info(
        "\n=== Exp-B: %s (seed %d) -- %d studi kasus BERSIH (aspek bernama + koheren) ===\n%s",
        domain, seed, len(out), out.to_string(index=False),
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
