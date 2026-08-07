"""
scripts/analyze_agf_efficiency.py

Analisis EFISIENSI dari 60 run faktorial Tahap 7 -- tabel siap-manuskrip
berisi jumlah parameter, waktu latih, dan waktu prediksi per sel x domain.
Tidak menjalankan ulang apa pun: ketiga metrik sudah tercatat otomatis di
tiap `results_summary`.

=====================================================================
 BATASAN YANG WAJIB DINYATAKAN BILA ANGKA INI MASUK MANUSKRIP.
 Tanpa keempatnya, klaim efisiensi di sini MENYESATKAN.
=====================================================================
1. `train_time_seconds` mengukur **LAPISAN FUSI SAJA** (training AGF, atau
   fit NMF+DecisionTree) -- TIDAK termasuk DeepMF OOF 5-fold dan CBF LOO,
   yang justru merupakan biaya dominan pipeline. Angka ini sah sebagai
   klaim "biaya mekanisme fusi", BUKAN biaya end-to-end.
2. Sel A (A2-IRM) TIDAK PUNYA data waktu sama sekali -- YAML lamanya tidak
   merekam field timing. Jadi tabel ini TIDAK bisa dipakai mengklaim
   "arsitektur usulan lebih cepat/lambat dari A2-IRM". Yang bisa
   dibandingkan hanyalah antar sel B/C/D0/D.
3. `n_parameters` sel C bernilai None (tree tidak punya parameter dalam
   pengertian yang sama) -> kolom parameter tidak sebanding untuk sel itu;
   yang sebanding hanya waktu.
4. Diukur di Colab; hardware bisa berbeda antar sesi. Konsisten DALAM satu
   sesi, tapi perbandingan lintas-sesi tidak dijamin. Selain itu run
   memakai `--stream-cache`, TAPI status hit/miss cache TIDAK tercatat di
   `results_summary` -- lihat CATATAN di bawah.

CATATAN (utang teknis yang diketahui): penanda cache belum ada di
`results_summary`, sehingga waktu di sini tidak bisa difilter berdasar
apakah stream DeepMF/CBF dibangun atau dimuat dari cache. Ini TIDAK
memengaruhi `train_time_seconds` (yang hanya mencakup lapisan fusi,
dijalankan setelah stream siap), tapi berarti waktu total per-run tidak
tersedia secara bersih. Menambahkan penanda itu butuh perubahan runner dan
hanya berlaku untuk run BERIKUTNYA -- 60 run yang sudah ada tidak bisa
dilengkapi surut.

Usage:
    venv/Scripts/python.exe scripts/analyze_agf_efficiency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TEST_DIR = _REPO_ROOT / "checkpoints" / "results_phase2_clean" / "test"

DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant",
    "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456, 789, 1011]

CELLS = {
    "B": ("agf_agf_keyword_cellB", "B: keyword + AGF"),
    "C": ("agf_static_pyabsa_rich_cellC", "C: PyABSA-rich + tree"),
    "D0": ("agf_a2fusionrs_clean_cellD0", "D0: PyABSA-rich + AGF"),
    "D": ("agf_a2fusionrs_clean_cellD", "D: PyABSA seq+rich + AGF"),
}

METRICS = ["n_parameters", "train_time_seconds", "predict_time_seconds"]


def main() -> None:
    rows, missing = [], []
    for cell, (prefix, label) in CELLS.items():
        for domain in DOMAINS:
            vals: dict[str, list[float]] = {m: [] for m in METRICS}
            for seed in SEEDS:
                path = TEST_DIR / f"{prefix}_{domain}_seed{seed}.yaml"
                if not path.exists():
                    missing.append(path.name)
                    continue
                with open(path) as f:
                    data = yaml.safe_load(f)
                for m in METRICS:
                    v = data.get(m)
                    if v is not None:
                        vals[m].append(float(v))
            row = {"cell": cell, "cell_label": label, "domain": DOMAIN_LABELS[domain]}
            for m in METRICS:
                if vals[m]:
                    row[f"{m}_mean"] = float(np.mean(vals[m]))
                    row[f"{m}_std"] = float(np.std(vals[m], ddof=1)) if len(vals[m]) > 1 else 0.0
                    row[f"{m}_n"] = len(vals[m])
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan
                    row[f"{m}_n"] = 0
            rows.append(row)

    df = pd.DataFrame(rows)
    if missing:
        print(f"!! {len(missing)} file hilang: {missing[:5]}{' ...' if len(missing) > 5 else ''}\n")

    pd.set_option("display.width", 220)

    print("=" * 78)
    print("EFISIENSI LAPISAN FUSI (mean +/- SD lintas 5 seed, 60 run Tahap 7)")
    print("=" * 78)
    print("\n-- Waktu latih lapisan fusi (detik) --")
    print(df.pivot_table(index=["cell", "cell_label"], columns="domain",
                         values="train_time_seconds_mean").round(2).to_string())
    print("\n-- Waktu prediksi (detik, seluruh test set) --")
    print(df.pivot_table(index=["cell", "cell_label"], columns="domain",
                         values="predict_time_seconds_mean").round(4).to_string())
    print("\n-- Jumlah parameter (sel C = tree -> tidak sebanding) --")
    par = df.pivot_table(index=["cell", "cell_label"], columns="domain",
                         values="n_parameters_mean")
    print(par.round(0).to_string())

    n_par_missing = int((df["n_parameters_n"] == 0).sum())
    if n_par_missing:
        print(f"\n   ({n_par_missing} sel x domain tanpa n_parameters -- sel C memakai tree.)")

    dest = TEST_DIR / "efficiency_summary.csv"
    df.to_csv(dest, index=False)

    print("\n" + "=" * 78)
    print("BATASAN WAJIB DINYATAKAN DI MANUSKRIP")
    print("=" * 78)
    print(
        "1. Waktu latih = LAPISAN FUSI SAJA; TIDAK termasuk DeepMF OOF 5-fold\n"
        "   & CBF LOO yang merupakan biaya dominan. Sah sbg 'biaya mekanisme\n"
        "   fusi', BUKAN biaya end-to-end.\n"
        "2. Sel A (A2-IRM) TIDAK punya data waktu -> tabel ini TIDAK bisa\n"
        "   dipakai mengklaim usulan lebih cepat/lambat dari A2-IRM.\n"
        "3. Sel C (tree) tidak punya n_parameters yang sebanding dgn AGF.\n"
        "4. Diukur di Colab; konsisten dalam satu sesi, tidak lintas sesi.\n"
        "   Status hit/miss stream-cache tidak tercatat (utang teknis)."
    )
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
