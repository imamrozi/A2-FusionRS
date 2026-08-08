"""
scripts/retest_structure_recovery.py

UJI ULANG pemulihan struktur, MEMPERBAIKI CACAT ENCODING pada gerbang-2.

CACAT YANG DIPERBAIKI
Di `gate_pyabsa_extraction_sabert_scoring.py`, representasi berstruktur
(`clustered`, `bag200`) mengisi kolom skor dengan **0.0** untuk cluster/
istilah yang tidak muncul di sebuah review. Pada skala sentimen [0,1],
0.0 berarti "SANGAT NEGATIF", bukan "tidak ada data". Jadi setiap aspek
yang absen disuntikkan sebagai sinyal negatif palsu yang kuat.

Ini membuat perbandingannya TIDAK ADIL terhadap PyABSA: `keyword_concat_conf`
tidak punya masalah ini karena aspek yang tidak match diberi skor fallback
seluruh review (lihat absa_bert.py), bukan nol. Karena aspek keyword rapat
(4-6 kolom, hampir selalu terisi) sedangkan aspek PyABSA jarang (~200
istilah, tiap review hanya punya sedikit), cacat ini menghukum cabang
PyABSA jauh lebih berat.

Skrip ini menguji ulang dgn tiga skema encoding, memakai CACHE skor
per-aspek SA-BERT (`sabert_aspect_scores_gatesample.csv`) sehingga TIDAK
perlu mengulang inference BERT ~40 menit/domain:

  v0_zero      : encoding lama (0.0 utk absen) -- direproduksi sbg kontrol,
                 harus cocok dgn angka gerbang-2.
  v1_impute    : absen -> diisi rata-rata level-review (analog persis
                 perlakuan keyword: fallback seluruh review).
  v2_impute_ind: v1 + indikator kehadiran eksplisit per slot, sehingga
                 model bisa membedakan "diimputasi" dari "benar-benar
                 terukur" -- praktik standar utk fitur missing.

Pembanding tetap: keyword_concat_conf (palang absolut) dan
pyabsaext_sabert_rich9 (usulan tanpa struktur).

Usage:
    venv/Scripts/python.exe scripts/retest_structure_recovery.py
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

from diagnose_sentiment_signal_quality import _cluster_aspect_terms  # noqa: E402
from gate_pyabsa_extraction_sabert_scoring import (  # noqa: E402
    DOMAINS, DOMAIN_LABELS, N_TEST, N_TRAIN, RATING_COL, RIDGE_ALPHA,
    SABERT_RICH_NAMES, SAMPLE_SEED, _keyword_rich, _sabert_rich,
)
from src.a2fusionrs.pyabsa_scorer import load_cached_scores  # noqa: E402

TOP_K = 200


def _fit_eval(x_tr, y_tr, x_te, y_te) -> dict:
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x_tr), y_tr)
    pred = model.predict(scaler.transform(x_te))
    return {
        "rmse": float(np.sqrt(np.mean((pred - y_te) ** 2))),
        "pearson_r": float(np.corrcoef(pred, y_te)[0, 1]) if np.std(pred) > 1e-12 else 0.0,
        "n_features": x_tr.shape[1],
    }


def main() -> None:
    rows = []

    for domain, cfg in DOMAINS.items():
        print(f"\n{'=' * 72}\n{DOMAIN_LABELS[domain]}\n{'=' * 72}")

        base = _REPO_ROOT / "data" / "splits" / cfg["split"]
        train = pd.read_csv(base / "train.csv", usecols=["review_id", RATING_COL])
        test = pd.read_csv(base / "test.csv", usecols=["review_id", RATING_COL])
        train = train.sample(n=min(N_TRAIN, len(train)), random_state=SAMPLE_SEED)
        test = test.sample(n=min(N_TEST, len(test)), random_state=SAMPLE_SEED)
        ids = pd.concat([train, test], ignore_index=True)["review_id"].tolist()
        id_pos = {r: i for i, r in enumerate(ids)}

        pdir = _REPO_ROOT / "checkpoints" / cfg["ckpt"] / "pyabsa"
        asp = pd.read_csv(pdir / "sabert_aspect_scores_gatesample.csv")
        fb = pd.read_csv(pdir / "sabert_fallback_gatesample.csv")
        fallback = dict(zip(fb["review_id"], fb["fallback_score"]))

        term_scores: dict[int, list[tuple[str, float]]] = {}
        for rid, term, s in zip(asp["review_id"], asp["aspect_term"], asp["sabert_score"]):
            term_scores.setdefault(rid, []).append((str(term), float(s)))

        # nilai imputasi = rata-rata aspek yang ADA di review itu; kalau
        # review tak punya aspek sama sekali -> skor fallback seluruh review.
        # Ini analog PERSIS dgn perlakuan keyword-ABSA thd aspek tak-match.
        impute = np.full(len(ids), 0.5, dtype=np.float32)
        for rid, pos in id_pos.items():
            pairs = term_scores.get(rid)
            impute[pos] = (
                float(np.mean([s for _, s in pairs])) if pairs else fallback.get(rid, 0.5)
            )

        sdir = _REPO_ROOT / "checkpoints" / cfg["ckpt"] / "sentiment_bert"
        kw = pd.read_csv(sdir / "absa_concat_confidence_scores.csv")
        n_kw = (len(kw.columns) - 1) // 2

        train_ids = set(train["review_id"])
        counter: dict[str, int] = {}
        for rid, pairs in term_scores.items():
            if rid in train_ids:
                for term, _ in pairs:
                    counter[term] = counter.get(term, 0) + 1
        vocab = [t for t, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:TOP_K]]
        vidx = {t: i for i, t in enumerate(vocab)}

        scored = load_cached_scores(str(pdir / f"pyabsa_scores_{cfg['pyabsa']}.csv"))
        scored = scored[scored["review_id"].isin(set(ids))].reset_index(drop=True)
        term2cluster = _cluster_aspect_terms(scored, train_ids, vocab, n_kw)

        # ---- rakit tiga skema encoding utk clustered & bag ----
        def build(slots: int, slot_of, encoding: str) -> np.ndarray:
            width = slots * (3 if encoding == "v2_impute_ind" else 2)
            out = np.zeros((len(ids), width), dtype=np.float32)
            for rid, pos in id_pos.items():
                acc: dict[int, list[float]] = {}
                for term, s in term_scores.get(rid, []):
                    j = slot_of(term)
                    if j is not None:
                        acc.setdefault(j, []).append(s)
                for j in range(slots):
                    present = j in acc
                    if present:
                        out[pos, j] = float(np.mean(acc[j]))
                    elif encoding == "v0_zero":
                        out[pos, j] = 0.0          # CACAT: 0 = "sangat negatif"
                    else:
                        out[pos, j] = impute[pos]  # PERBAIKAN: analog fallback keyword
                    out[pos, slots + j] = 1.0 if present else 0.0
                    if encoding == "v2_impute_ind":
                        out[pos, 2 * slots + j] = 0.0 if present else 1.0
            return out

        rich = pd.DataFrame(
            _sabert_rich(
                {r: [s for _, s in term_scores.get(r, [])] for r in ids}, ids, fallback
            ),
            columns=SABERT_RICH_NAMES,
        )
        rich.insert(0, "review_id", ids)

        reps: dict[str, pd.DataFrame] = {
            "keyword_concat_conf (palang absolut)": kw,
            "keyword_rich9 (kontrol)": _keyword_rich(kw),
            "pyabsaext_sabert_rich9 (usulan)": rich,
        }
        for enc in ("v0_zero", "v1_impute", "v2_impute_ind"):
            for name, slots, fn in (
                ("clustered", n_kw, lambda t: term2cluster.get(t)),
                ("bag200", len(vocab), lambda t: vidx.get(t)),
            ):
                arr = build(slots, fn, enc)
                df = pd.DataFrame(arr, columns=[f"{name}_{enc}_{i}" for i in range(arr.shape[1])])
                df.insert(0, "review_id", ids)
                reps[f"{name} [{enc}]"] = df

        for label, feat in reps.items():
            fcols = [c for c in feat.columns if c != "review_id"]
            tr = train.merge(feat, on="review_id", how="inner")
            te = test.merge(feat, on="review_id", how="inner")
            if tr.empty or te.empty:
                continue
            res = _fit_eval(
                tr[fcols].to_numpy(np.float64), tr[RATING_COL].to_numpy(np.float64),
                te[fcols].to_numpy(np.float64), te[RATING_COL].to_numpy(np.float64),
            )
            rows.append({"domain": DOMAIN_LABELS[domain], "representation": label, **res})
            print(f"  {label:<40} dim={res['n_features']:>4}  RMSE={res['rmse']:.4f}")

    out = pd.DataFrame(rows)
    dest = _REPO_ROOT / "reports" / "structure_recovery_retest.csv"
    out.to_csv(dest, index=False)

    print(f"\n\n{'=' * 72}\nRINGKASAN (RMSE kanal sentimen saja)\n{'=' * 72}")
    print(out.pivot_table(index="representation", columns="domain", values="rmse").to_string())
    print(f"\nDisimpan ke {dest}")
    print(
        "\nCARA MEMBACA:\n"
        "  v0_zero harus mereproduksi angka gerbang-2 (kontrol kebenaran).\n"
        "  Jika v1/v2 jauh lebih baik dari v0 -> kesimpulan gerbang-2 memang\n"
        "  tercemar cacat encoding, dan pemulihan struktur perlu dinilai ulang.\n"
        "  Jika v1/v2 tetap kalah dari 'usulan rich9' -> struktur posisi-tetap\n"
        "  memang TIDAK dapat dipulihkan dari aspek open-vocabulary, dan itu\n"
        "  temuan yang sah (bukan artefak encoding)."
    )


if __name__ == "__main__":
    main()
