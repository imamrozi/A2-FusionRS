"""
scripts/analyze_agf_v2_selection.py

Menerapkan ATURAN KEPUTUSAN PRA-REGISTRASI dari
`scripts/run_agf_v2_selection.sh` -- seleksi kapasitas AGF di SELECTION_DEV
untuk arsitektur v2 (ekstraksi PyABSA + skoring SA-BERT + token global).

Aturan (disalin verbatim dari header script, JANGAN diubah setelah melihat
hasil):
  1. Per domain: rata-rata dev-RMSE lintas 3 seed -> ranking 6 varian.
  2. Pemenang = rata-rata rank TERENDAH lintas 3 domain.
  3. Tie-break: mean RMSE ternormalisasi (dibagi RMSE terbaik per domain).
  4. SATU konfigurasi untuk ketiga domain -- TIDAK boleh per-domain.
  5. Tabel seleksi WAJIB di-commit SEBELUM konfirmasi di test dijalankan.

PENGAMAN (menolak jalan, bukan diam-diam menghasilkan angka salah):
  - Menolak file dengan `eval_split != "selection_dev"`. Memilih arsitektur
    berdasarkan test set adalah p-hacking langsung -- itulah cacat yang
    memicu seluruh protokol ini (lihat plan pure-painting-wilkes.md).
  - Menolak melaporkan pemenang kalau ada sel yang tidak lengkap; ranking
    dari data bolong tidak dapat dipertanggungjawabkan.
  - Memverifikasi setiap file benar-benar memakai konfigurasi arsitektur
    yang diklaim konstan (extra_pyabsa, global token, jangkar, representasi).

Usage:
    python scripts/analyze_agf_v2_selection.py
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

DEV_DIR = _REPO_ROOT / "checkpoints" / "results_phase2_clean_v2" / "dev"

DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant",
    "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456]

AGF_D = [64, 128]
AGF_WD = [0.0, 0.0001, 0.001]

# Konfigurasi yang HARUS konstan di seluruh grid -- diverifikasi per file.
EXPECTED_CONSTANT = {
    "scenario": "a2fusionrs_clean",
    "extra_pyabsa": "sabert_perseq_rich",
    "global_sentiment_token": True,
    "residual_base": "user_item_bias",
    "representation": "asymmetric",
    "input_standardize": True,
}


def _tag(d: int, wd: float) -> str:
    """Meniru pembentukan tag di run_agf_v2_selection.sh: hapus '.' dan '-'."""
    return f"d{d}wd" + str(wd).replace(".", "").replace("-", "")


def main() -> None:
    rows, missing, violations = [], [], []

    for d in AGF_D:
        for wd in AGF_WD:
            tag = _tag(d, wd)
            for domain in DOMAINS:
                for seed in SEEDS:
                    path = DEV_DIR / f"agf_a2fusionrs_clean_{tag}_{domain}_seed{seed}.yaml"
                    if not path.exists():
                        missing.append(path.name)
                        continue
                    with open(path) as f:
                        data = yaml.safe_load(f)

                    if data.get("eval_split") != "selection_dev":
                        raise SystemExit(
                            f"BERHENTI: {path.name} punya eval_split="
                            f"'{data.get('eval_split')}', bukan 'selection_dev'. "
                            "Memilih arsitektur berdasarkan test set adalah p-hacking."
                        )
                    for key, want in EXPECTED_CONSTANT.items():
                        if data.get(key) != want:
                            violations.append(
                                f"{path.name}: {key}={data.get(key)!r}, seharusnya {want!r}"
                            )
                    if data.get("agf_d") != d or not np.isclose(
                        float(data.get("agf_weight_decay", -1)), wd
                    ):
                        violations.append(
                            f"{path.name}: agf_d={data.get('agf_d')} / "
                            f"weight_decay={data.get('agf_weight_decay')} tidak cocok "
                            f"dengan tag '{tag}'"
                        )

                    rows.append({
                        "tag": tag, "d": d, "weight_decay": wd,
                        "domain": DOMAIN_LABELS[domain], "seed": seed,
                        "dev_rmse": data["rmse"], "val_rmse": data.get("val_rmse"),
                        "n_parameters": data.get("n_parameters"),
                        "train_time_seconds": data.get("train_time_seconds"),
                    })

    if violations:
        print(f"!! {len(violations)} pelanggaran konfigurasi konstan:")
        for v in violations[:10]:
            print("   -", v)
        raise SystemExit(
            "BERHENTI: grid tidak apple-to-apple. Perbandingan tidak sah."
        )

    if not rows:
        raise SystemExit(f"Tidak ada hasil di {DEV_DIR}. Jalankan run_agf_v2_selection.sh dulu.")

    df = pd.DataFrame(rows)
    raw_dest = _REPO_ROOT / "reports" / "agf_v2_selection"
    raw_dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dest / "raw_per_run_dev.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    # --- Langkah 1: mean dev-RMSE per (varian, domain) ---
    mean_rmse = df.groupby(["tag", "domain"])["dev_rmse"].mean().unstack()
    print("=" * 78)
    print("Mean dev-RMSE lintas 3 seed (SELECTION_DEV -- test TIDAK disentuh)")
    print("=" * 78)
    print(mean_rmse.to_string())

    n_expected = len(AGF_D) * len(AGF_WD) * len(DOMAINS) * len(SEEDS)
    if missing:
        print(f"\n!! {len(missing)}/{n_expected} run belum ada:")
        for m in missing[:8]:
            print("   -", m)
        raise SystemExit(
            "\nPEMENANG TIDAK DILAPORKAN sampai grid lengkap -- ranking dari data "
            "bolong tidak dapat dipertanggungjawabkan. Jalankan ulang "
            "run_agf_v2_selection.sh (resumable)."
        )

    # --- Langkah 2: ranking per domain, lalu mean rank ---
    ranks = mean_rmse.rank(axis=0, method="average")
    mean_rank = ranks.mean(axis=1).sort_values()

    # --- Langkah 3: tie-break = mean RMSE ternormalisasi ---
    normed = (mean_rmse / mean_rmse.min(axis=0)).mean(axis=1)

    summary = pd.DataFrame({
        "mean_rank": mean_rank,
        "normalized_rmse": normed.reindex(mean_rank.index),
    })
    summary = summary.sort_values(["mean_rank", "normalized_rmse"])

    print("\n" + "=" * 78)
    print("RANKING (aturan pra-registrasi: mean rank, tie-break RMSE ternormalisasi)")
    print("=" * 78)
    print(summary.to_string())

    winner = summary.index[0]
    w = df[df["tag"] == winner].iloc[0]
    tied = summary[np.isclose(summary["mean_rank"], summary["mean_rank"].iloc[0])]

    print("\n" + "=" * 78)
    print(f"PEMENANG: {winner}  (d={w['d']}, weight_decay={w['weight_decay']})")
    print("=" * 78)
    if len(tied) > 1:
        gap = summary["normalized_rmse"].iloc[1] - summary["normalized_rmse"].iloc[0]
        print(
            f"CATATAN KEJUJURAN: {len(tied)} varian SERI pada mean rank "
            f"({summary['mean_rank'].iloc[0]:.2f}); pemenang ditentukan tie-break "
            f"dgn selisih RMSE ternormalisasi hanya {gap:.4f} "
            f"({gap * 100:.2f}%). Perbedaan sekecil ini WAJIB dinyatakan di "
            "manuskrip -- jangan diklaim sebagai keunggulan yang berarti."
        )

    # Diagnostik untuk target ROBUSTNESS: SD lintas seed per varian.
    sd = df.groupby(["tag", "domain"])["dev_rmse"].std(ddof=1).unstack()
    print("\n" + "-" * 78)
    print("SD dev-RMSE lintas seed (diagnostik ROBUSTNESS -- BUKAN kriteria seleksi)")
    print("-" * 78)
    print(sd.to_string())
    print(
        "\nTabel SD ini TIDAK dipakai memilih pemenang (aturan pra-registrasi\n"
        "memakai mean RMSE). Ia dilaporkan karena robustness adalah target\n"
        "kedua; klaim robustness yang sah harus diukur di TEST, bukan di dev."
    )

    summary.to_csv(raw_dest / "selection_ranking.csv")
    mean_rmse.to_csv(raw_dest / "mean_dev_rmse.csv")
    sd.to_csv(raw_dest / "sd_dev_rmse.csv")
    print(f"\nDisimpan ke {raw_dest}/")
    print(
        "\nLANGKAH BERIKUTNYA (WAJIB, urutannya penting):\n"
        f"  1. COMMIT {raw_dest.name}/ -- jejak audit bahwa pemenang ditetapkan\n"
        "     SEBELUM test set disentuh.\n"
        f"  2. Kunci pemenang di scripts/run_agf_v2_factorial.sh (d={w['d']}, "
        f"weight_decay={w['weight_decay']}).\n"
        "  3. Baru jalankan konfirmasi di test."
    )


if __name__ == "__main__":
    main()
