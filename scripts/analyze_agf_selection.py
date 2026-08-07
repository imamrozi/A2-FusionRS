"""
scripts/analyze_agf_selection.py

Tahap 6 (plan pure-painting-wilkes.md): tentukan varian jangkar PEMENANG
dari hasil seleksi DEV (scripts/run_agf_clean_selection.sh).

ATURAN KEPUTUSAN -- DIPRA-REGISTRASI SEBELUM RUN DIJALANKAN, diimplemen-
tasikan di sini APA ADANYA (lihat header run_agf_clean_selection.sh):
  1. Per domain: rata-rata dev-RMSE lintas seed untuk tiap varian.
  2. Ranking 4 varian per domain (1 = terbaik/RMSE terendah).
  3. Varian dgn RATA-RATA RANKING terendah lintas domain = PEMENANG.
  4. Tie-break: mean dev-RMSE ternormalisasi per domain
     (RMSE_varian / RMSE_terbaik_di_domain_itu), dirata-rata lintas domain.
  5. SATU konfigurasi untuk ketiga domain -- pemilihan per-domain DILARANG
     (overfitting ke dev).

PENGAMAN AKADEMIK:
- Script ini HANYA membaca hasil ber-`stage: select` / `eval_split:
  selection_dev`. Kalau menemukan file ber-`eval_split: test`, ia BERHENTI
  dengan error -- seleksi arsitektur TIDAK BOLEH menyentuh test set.
- Kelengkapan diverifikasi: bila ada sel (domain x seed x varian) yang
  hilang, dilaporkan eksplisit dan pemenang TIDAK ditetapkan (keputusan
  atas data tidak lengkap tidak bisa dipertanggungjawabkan).

Usage:
    python scripts/analyze_agf_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEV_DIR = _REPO_ROOT / "checkpoints" / "results_phase2_clean" / "dev"
DOMAINS = ["restaurant", "amazon_electronics", "tripadvisor_hotel"]
DOMAIN_LABELS = {
    "restaurant": "Restaurant", "amazon_electronics": "E-commerce", "tripadvisor_hotel": "Hotel",
}
SEEDS = [42, 123, 456]
VARIANTS = [  # (run_tag, label terbaca)
    ("none_vector", "tanpa jangkar + vektor"),
    ("none_asymmetric", "tanpa jangkar + skalar"),
    ("user_item_bias_vector", "jangkar bias + vektor"),
    ("user_item_bias_asymmetric", "jangkar bias + skalar"),
]


def load_dev_results() -> tuple[pd.DataFrame, list[str]]:
    rows, missing = [], []
    for domain in DOMAINS:
        for tag, label in VARIANTS:
            for seed in SEEDS:
                path = DEV_DIR / f"agf_a2fusionrs_clean_{tag}_{domain}_seed{seed}.yaml"
                if not path.exists():
                    missing.append(path.name)
                    continue
                with open(path) as f:
                    d = yaml.safe_load(f)

                eval_split = d.get("eval_split")
                if eval_split != "selection_dev":
                    raise SystemExit(
                        f"BERHENTI: {path.name} punya eval_split='{eval_split}', bukan "
                        "'selection_dev'. Seleksi arsitektur TIDAK BOLEH memakai hasil test set."
                    )
                rows.append({
                    "domain": domain, "variant": tag, "variant_label": label,
                    "seed": seed, "dev_rmse": d["rmse"], "val_rmse": d.get("val_rmse"),
                })
    return pd.DataFrame(rows), missing


def main() -> None:
    df, missing = load_dev_results()
    if df.empty:
        raise SystemExit(
            f"Tidak ada hasil seleksi di {DEV_DIR}. Jalankan "
            "scripts/run_agf_clean_selection.sh terlebih dahulu."
        )

    expected = len(DOMAINS) * len(VARIANTS) * len(SEEDS)
    print(f"Hasil dimuat: {len(df)}/{expected} sel\n")

    # ---- Langkah 1: mean dev-RMSE per (domain, varian) ----
    per_domain = (
        df.groupby(["domain", "variant", "variant_label"])["dev_rmse"]
        .agg(["mean", "std", "count"]).reset_index()
        .rename(columns={"mean": "dev_rmse_mean", "std": "dev_rmse_std", "count": "n_seeds"})
    )

    # ---- Langkah 2: ranking per domain (1 = terbaik) ----
    per_domain["rank"] = per_domain.groupby("domain")["dev_rmse_mean"].rank(method="min")
    # ---- Tie-break: RMSE ternormalisasi thd varian terbaik di domain itu ----
    per_domain["rmse_normalized"] = per_domain.groupby("domain")["dev_rmse_mean"].transform(
        lambda s: s / s.min()
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("=== dev-RMSE per domain x varian (mean +/- SD lintas seed) ===")
    for domain in DOMAINS:
        sub = per_domain[per_domain["domain"] == domain].sort_values("rank")
        print(f"\n-- {DOMAIN_LABELS[domain]} --")
        print(sub[["variant_label", "dev_rmse_mean", "dev_rmse_std", "n_seeds", "rank"]]
              .to_string(index=False))

    # ---- Langkah 3+4: agregasi ranking lintas domain ----
    overall = (
        per_domain.groupby(["variant", "variant_label"])
        .agg(mean_rank=("rank", "mean"),
             mean_rmse_normalized=("rmse_normalized", "mean"),
             n_domains=("domain", "nunique"))
        .reset_index()
        .sort_values(["mean_rank", "mean_rmse_normalized"])
    )

    print("\n\n=== AGREGASI LINTAS DOMAIN (aturan keputusan pra-registrasi) ===")
    print(overall.to_string(index=False))

    out_path = DEV_DIR / "selection_table.csv"
    per_domain.to_csv(out_path, index=False)
    overall.to_csv(DEV_DIR / "selection_ranking.csv", index=False)
    print(f"\nTabel disimpan ke {out_path} dan selection_ranking.csv")

    if missing:
        print(f"\n!! {len(missing)} sel HILANG -- pemenang TIDAK ditetapkan.")
        for m in missing[:10]:
            print("   -", m)
        if len(missing) > 10:
            print(f"   ... dan {len(missing) - 10} lainnya")
        raise SystemExit(
            "Keputusan atas data tidak lengkap tidak bisa dipertanggungjawabkan. "
            "Jalankan ulang scripts/run_agf_clean_selection.sh (resumable) sampai lengkap."
        )

    winner = overall.iloc[0]
    res, rep = winner["variant"].rsplit("_", 1)
    print("\n" + "=" * 70)
    print(f"PEMENANG: {winner['variant_label']}  (--residual-base {res} --representation {rep})")
    print(f"  mean rank lintas domain = {winner['mean_rank']:.2f}")
    print(f"  mean dev-RMSE ternormalisasi = {winner['mean_rmse_normalized']:.4f}")
    print("=" * 70)
    print(
        "\nLANGKAH BERIKUTNYA: commit tabel seleksi ini DULU (jejak audit bahwa pemenang\n"
        "ditetapkan SEBELUM test disentuh), baru jalankan Tahap 7 (faktorial di test,\n"
        "--stage confirm)."
    )


if __name__ == "__main__":
    main()
