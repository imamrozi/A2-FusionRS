"""
scripts/gate3_global_sentiment_token.py

GERBANG-3: menemukan & menguji sumber sebenarnya dari "keunggulan struktur".

TEMUAN YANG DIUJI DI SINI
Gerbang-1/2 menyimpulkan bahwa `keyword_concat_conf` unggul 5,7-11,6% atas
bentuk order-statistics, dan menamai selisih itu "nilai struktur
posisi-tetap". Skrip ini menunjukkan label itu SALAH.

Mekanismenya: pada keyword-ABSA, aspek yang TIDAK match diisi skor sentimen
SELURUH REVIEW (fallback, lihat absa_bert.py). Karena banyak review hanya
mencocokkan sebagian aspek, sebagian besar kolom `keyword_concat_conf`
sebenarnya memuat skor review global -- jadi model selalu mendapat akses
LANGSUNG ke sinyal global itu secara cuma-cuma. Ringkasan order-statistics
(min/max/range/mean) menghancurkan akses tersebut: rata-rata campuran skor
aspek dan fallback bukanlah skor review itu sendiri.

Prediksi yang diuji: tambahkan skor review global sbg FITUR EKSPLISIT, dan
selisih "struktur" harus hilang. Hasilnya bahkan lebih kuat -- order-
statistics + global MENGUNGGULI concat posisi-tetap.

KEJUJURAN YANG WAJIB DIPERTAHANKAN DI MANUSKRIP: perbaikan ini menolong
KEDUA cabang. Ia BUKAN keunggulan PyABSA, melainkan fitur yang hilang dari
representasi gaya A2-IRM. Karena itu pembanding yang sah untuk usulan
adalah `kw_concat + global`, BUKAN `kw_concat` polos. Melaporkan hanya
lawan `kw_concat` polos akan melebih-lebihkan kontribusi PyABSA ~2x.

Varian `usul + kwc + global` disertakan sbg BATAS ATAS informasi saja --
ia memakai DUA sistem ABSA sekaligus ("ABSA 2x") yang sudah ditolak
sebagai arsitektur, jadi TIDAK boleh diklaim sebagai model usulan.

BATASAN: ini probe LINIER (Ridge) pada KANAL SENTIMEN SAJA, di subsample
5000 train / 3000 test. Angkanya tidak sebanding dgn RMSE rekomendasi
end-to-end, dan keunggulan ~1% di sini belum tentu bertahan setelah
digabung DeepMF+CBF yang mendominasi varians. Yang sah disimpulkan hanya
PERINGKAT ANTAR-REPRESENTASI.

Usage:
    venv/Scripts/python.exe scripts/gate3_global_sentiment_token.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gate_pyabsa_extraction_sabert_scoring import (  # noqa: E402
    DOMAINS, DOMAIN_LABELS, N_TEST, N_TRAIN, RATING_COL, RIDGE_ALPHA,
    SABERT_RICH_NAMES, SAMPLE_SEED, _keyword_rich, _sabert_rich,
)


def _fit_eval(tr: pd.DataFrame, te: pd.DataFrame, fcols: list[str]) -> float:
    scaler = StandardScaler().fit(tr[fcols])
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(tr[fcols]), tr[RATING_COL])
    pred = model.predict(scaler.transform(te[fcols]))
    return float(np.sqrt(np.mean((pred - te[RATING_COL]) ** 2)))


def main() -> None:
    rows = []

    for domain, cfg in DOMAINS.items():
        base = _REPO_ROOT / "data" / "splits" / cfg["split"]
        train = pd.read_csv(base / "train.csv", usecols=["review_id", RATING_COL]).sample(
            n=N_TRAIN, random_state=SAMPLE_SEED
        )
        test = pd.read_csv(base / "test.csv", usecols=["review_id", RATING_COL]).sample(
            n=N_TEST, random_state=SAMPLE_SEED
        )
        ids = pd.concat([train, test], ignore_index=True)["review_id"].tolist()

        ck = _REPO_ROOT / "checkpoints" / cfg["ckpt"]
        kw = pd.read_csv(ck / "sentiment_bert" / "absa_concat_confidence_scores.csv")
        glob = pd.read_csv(ck / "sentiment_bert" / "sentiment_scores.csv")

        asp = pd.read_csv(ck / "pyabsa" / "sabert_aspect_scores_gatesample.csv")
        fbdf = pd.read_csv(ck / "pyabsa" / "sabert_fallback_gatesample.csv")
        fallback = dict(zip(fbdf["review_id"], fbdf["fallback_score"]))
        per_review: dict[int, list[float]] = {}
        for rid, s in zip(asp["review_id"], asp["sabert_score"]):
            per_review.setdefault(rid, []).append(float(s))

        usul = pd.DataFrame(
            _sabert_rich(per_review, ids, fallback), columns=SABERT_RICH_NAMES
        )
        usul.insert(0, "review_id", ids)
        kwr = _keyword_rich(kw)

        variants = {
            "kw_concat (A2-IRM apa adanya)": kw,
            "kw_rich9 (order-stats)": kwr,
            "kw_concat + global": kw.merge(glob, on="review_id"),
            "kw_rich9 + global": kwr.merge(glob, on="review_id"),
            "usulan_rich9 (PyABSA+SA-BERT)": usul,
            "USULAN usulan_rich9 + global": usul.merge(glob, on="review_id"),
            "[batas atas] usulan + kw_concat + global":
                usul.merge(glob, on="review_id").merge(kw, on="review_id"),
        }

        print(f"\n{'=' * 72}\n{DOMAIN_LABELS[domain]}\n{'=' * 72}")
        for label, feat in variants.items():
            fcols = [c for c in feat.columns if c != "review_id"]
            tr = train.merge(feat, on="review_id", how="inner")
            te = test.merge(feat, on="review_id", how="inner")
            rmse = _fit_eval(tr, te, fcols)
            rows.append({
                "domain": DOMAIN_LABELS[domain], "representation": label,
                "n_features": len(fcols), "rmse": rmse,
            })
            print(f"  {label:<42} dim={len(fcols):>4}  RMSE={rmse:.4f}")

    out = pd.DataFrame(rows)
    dest = _REPO_ROOT / "reports" / "gate3_global_token.csv"
    out.to_csv(dest, index=False)

    piv = out.pivot_table(index="representation", columns="domain", values="rmse")
    print(f"\n\n{'=' * 72}\nRINGKASAN (RMSE kanal sentimen saja, subsample)\n{'=' * 72}")
    print(piv.to_string())

    print(f"\n{'=' * 72}\nPERBANDINGAN YANG SAH: usulan+global vs kw_concat+global\n{'=' * 72}")
    fair, naive = "kw_concat + global", "kw_concat (A2-IRM apa adanya)"
    usul = "USULAN usulan_rich9 + global"
    n_win = 0
    for dom in piv.columns:
        u, f, n = piv.loc[usul, dom], piv.loc[fair, dom], piv.loc[naive, dom]
        n_win += u <= f
        print(f"  {dom:12} usulan={u:.4f}  vs adil={f:.4f} ({(u-f)/f*100:+.1f}%)"
              f"   [vs naif={n:.4f} ({(u-n)/n*100:+.1f}%)]")
    print(f"\n  Usulan menang di {n_win}/{len(piv.columns)} domain pada pembanding ADIL.")
    print(
        "\n  CATATAN WAJIB: selisih thd pembanding NAIF jauh lebih besar, tapi\n"
        "  sebagian besarnya berasal dari token global yg menolong KEDUA cabang.\n"
        "  Melaporkan angka naif sbg kontribusi PyABSA = melebih-lebihkan ~2x."
    )
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
