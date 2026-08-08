"""
scripts/analyze_agf_robustness.py

ANALISIS ROBUSTNESS -- target kedua proyek, di samping RMSE rata-rata.

MOTIVASI (bukti yang sudah ada, faktorial v1 60 run): pada e-commerce,
A2-IRM KOLAPS di seed 1011 (RMSE 0,8961 vs ~0,67 di empat seed lain)
sementara AGF hanya naik ke 0,7758. Akibatnya SD lintas seed A2-IRM 0,1023
vs AGF 0,0481. Keunggulan "rata-rata" AGF di domain itu (-3,3%) sebenarnya
HAMPIR SELURUHNYA berasal dari satu seed sulit: buang seed 1011 dan AGF
justru +0,1% (praktis seri).

Pembacaan yang jujur: AGF bukan lebih AKURAT, melainkan lebih STABIL. Itu
klaim yang berbeda -- dan justru lebih kuat buktinya. Skrip ini mengukurnya
secara eksplisit alih-alih membiarkannya tersembunyi di dalam rata-rata.

METRIK YANG DILAPORKAN (semua di TEST, lintas seed):
  mean       rata-rata RMSE                     (akurasi tipikal)
  sd         simpangan baku RMSE lintas seed    (stabilitas)
  worst      RMSE terburuk lintas seed          (jaminan kasus terburuk)
  range      worst - best                       (rentang sensitivitas seed)
  cv         sd / mean                          (stabilitas ternormalisasi,
                                                 sebanding lintas domain)

BATASAN YANG WAJIB DINYATAKAN DI MANUSKRIP:
  - 5 seed adalah dasar yang TIPIS untuk estimasi varians. SD dari 5 sampel
    punya interval kepercayaan lebar; laporkan sebagai indikasi, JANGAN
    sebagai estimasi presisi. Kalau klaim robustness menjadi kontribusi
    utama, tambah jumlah seed (mis. 10-15) sebelum submit.
  - Variasi seed di sini mencakup split train/test DAN inisialisasi model
    sekaligus; keduanya tidak terpisah. Jadi ini mengukur "sensitivitas
    terhadap seed", bukan khusus "sensitivitas terhadap split".
  - Worst-case dari 5 seed adalah estimasi optimistik dari worst-case
    sebenarnya (minimum sampel selalu bias ke arah tidak-ekstrem).
  - Deteksi kolaps memakai z-score LEAVE-ONE-OUT. Ambang lazim
    "mean + 2*SD" MUSTAHIL terpicu pada sampel kecil: z maksimum yang
    mungkin adalah (n-1)/sqrt(n), yaitu 1,79 untuk n=5 -- outlier
    menaikkan SD-nya sendiri sehingga menutupi dirinya.

Usage:
    python scripts/analyze_agf_robustness.py                 # v2 (default)
    python scripts/analyze_agf_robustness.py --version v1    # faktorial lama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

A2IRM_DIR = _REPO_ROOT / "checkpoints" / "results"
A2IRM_PREFIX = "absa_ablation_concat_confidence_cbf_nosentiment"

DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant", "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456, 789, 1011]

VERSIONS = {
    "v2": {
        "dir": _REPO_ROOT / "checkpoints" / "results_phase2_clean_v2" / "test",
        "cells": {
            "A":  (A2IRM_DIR, A2IRM_PREFIX, "A : keyword + tree (A2-IRM)"),
            "B'": (None, "agf_agf_keyword_cellBfair", "B': keyword + AGF + global"),
            "E":  (None, "agf_a2fusionrs_clean_cellE", "E : PyABSA + SA-BERT + AGF + global"),
            "F":  (None, "agf_a2fusionrs_clean_cellF", "F : + sequence  [TARGET]"),
            "F-": (None, "agf_a2fusionrs_clean_cellFminus", "F-: tanpa token global"),
        },
    },
    "v1": {
        "dir": _REPO_ROOT / "checkpoints" / "results_phase2_clean" / "test",
        "cells": {
            "A":  (A2IRM_DIR, A2IRM_PREFIX, "A : keyword + tree (A2-IRM)"),
            "B":  (None, "agf_agf_keyword_cellB", "B : keyword + AGF"),
            "C":  (None, "agf_static_pyabsa_rich_cellC", "C : PyABSA-rich + tree"),
            "D0": (None, "agf_a2fusionrs_clean_cellD0", "D0: PyABSA-rich + AGF"),
            "D":  (None, "agf_a2fusionrs_clean_cellD", "D : PyABSA seq+rich + AGF"),
        },
    },
}


def load_per_seed(base_dir: Path, cell_dir, prefix: str, domain: str) -> dict[int, float]:
    d = cell_dir if cell_dir is not None else base_dir
    out = {}
    for seed in SEEDS:
        path = d / f"{prefix}_{domain}_seed{seed}.yaml"
        if path.exists():
            with open(path) as f:
                out[seed] = yaml.safe_load(f)["rmse"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(VERSIONS), default="v2")
    args = ap.parse_args()
    spec = VERSIONS[args.version]

    rows, per_seed_rows = [], []
    for cell, (cell_dir, prefix, label) in spec["cells"].items():
        for domain in DOMAINS:
            vals = load_per_seed(spec["dir"], cell_dir, prefix, domain)
            if not vals:
                continue
            arr = np.array(list(vals.values()), dtype=float)
            for seed, v in vals.items():
                per_seed_rows.append({
                    "cell": cell, "domain": DOMAIN_LABELS[domain], "seed": seed, "rmse": v,
                })
            rows.append({
                "cell": cell, "cell_label": label, "domain": DOMAIN_LABELS[domain],
                "n_seeds": len(arr),
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "best": float(arr.min()), "worst": float(arr.max()),
                "range": float(arr.max() - arr.min()),
                "cv": float(arr.std(ddof=1) / arr.mean()) if len(arr) > 1 else 0.0,
            })

    if not rows:
        raise SystemExit(f"Tidak ada hasil di {spec['dir']}.")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 84)
    print(f"ROBUSTNESS ({args.version}) -- TEST SET, lintas {len(SEEDS)} seed")
    print("=" * 84)
    for metric, title in [
        ("mean", "Mean RMSE (akurasi tipikal)"),
        ("sd", "SD lintas seed (stabilitas -- makin kecil makin baik)"),
        ("worst", "RMSE TERBURUK lintas seed (jaminan kasus terburuk)"),
        ("cv", "Koefisien variasi = SD/mean (stabilitas ternormalisasi)"),
    ]:
        print(f"\n-- {title} --")
        print(df.pivot_table(index=["cell", "cell_label"], columns="domain",
                             values=metric).to_string())

    # Sorot seed yang menjadi outlier: bukti kolaps yang tersembunyi di mean.
    ps = pd.DataFrame(per_seed_rows)
    print("\n\n" + "=" * 84)
    print("DETEKSI KOLAPS: z-score LEAVE-ONE-OUT > 3 (seed dibandingkan seed LAIN)")
    print("=" * 84)
    print(
        "Memakai LOO, BUKAN 'mean + 2*SD' atas seluruh sampel: dgn n seed,\n"
        f"z-score maksimum yang MUNGKIN adalah (n-1)/sqrt(n) = "
        f"{(len(SEEDS) - 1) / np.sqrt(len(SEEDS)):.2f} untuk n={len(SEEDS)}, sehingga\n"
        "ambang 2*SD MUSTAHIL terpicu -- outlier menaikkan SD-nya sendiri.\n"
        "LOO menghindari kontaminasi itu dgn mengeluarkan seed yang diuji.\n"
    )
    found = False
    for (cell, domain), g in ps.groupby(["cell", "domain"]):
        if len(g) < 4:
            continue
        for _, r in g.iterrows():
            others = g[g["seed"] != r["seed"]]["rmse"]
            m, s = others.mean(), others.std(ddof=1)
            if s <= 0:
                continue
            z = (r["rmse"] - m) / s
            if z > 3:
                found = True
                print(f"  {cell:4} {domain:12} seed {int(r['seed']):>5}: "
                      f"RMSE={r['rmse']:.4f}  vs seed lain {m:.4f}+-{s:.4f}  "
                      f"-> z_LOO={z:.1f}")
    if not found:
        print("  (tidak ada seed dgn z_LOO > 3)")

    dest = spec["dir"] / f"robustness_{args.version}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    ps.to_csv(spec["dir"] / f"robustness_per_seed_{args.version}.csv", index=False)

    print("\n" + "=" * 84)
    print("BATASAN WAJIB DINYATAKAN DI MANUSKRIP")
    print("=" * 84)
    print(
        "1. 5 seed = dasar TIPIS untuk estimasi varians; SD dari 5 sampel punya\n"
        "   interval kepercayaan lebar. Kalau robustness jadi kontribusi utama,\n"
        "   tambah seed (10-15) sebelum submit.\n"
        "2. Variasi seed mencakup split DAN inisialisasi sekaligus -> ini\n"
        "   'sensitivitas terhadap seed', bukan khusus 'terhadap split'.\n"
        "3. Worst-case dari 5 seed adalah estimasi OPTIMISTIK dari worst-case\n"
        "   sebenarnya."
    )
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
