"""
scripts/diagnose_sentiment_signal_quality.py

DIAGNOSTIK LANJUTAN Tahap 7: MENGAPA PyABSA kalah dari keyword-ABSA?

Hasil faktorial menunjukkan penggantian keyword-ABSA -> PyABSA MERUGIKAN
+12,8%/+35,5%/+5,5% (fusi statis) dan +8,1%/+26,3%/+1,8% (AGF). Ada DUA
hipotesis bersaing untuk menjelaskannya:

  H-struktur : PyABSA open-vocabulary kehilangan STRUKTUR posisi-tetap yang
               dimiliki taksonomi keyword; agregasi/pooling merusak sinyal.
               -> kalau benar, memetakan PyABSA ke taksonomi tetap (mis.
                  lewat clustering istilah aspek) SEHARUSNYA menutup gap.
  H-supervisi: skor sentimen keyword-ABSA berasal dari `GlobalSentimentBERT`
               yang DI-FINE-TUNE pada label turunan `stars` domain itu
               sendiri (src/baseline/sentiment_bert.py:47
               `derive_sentiment_label`, dilatih di run_baseline.py:116 atas
               train+val). PyABSA memakai checkpoint pretrained generik
               TANPA supervisi rating apa pun.
               -> kalau benar, clustering TIDAK akan menolong, karena
                  defisitnya ada di KUALITAS skor, bukan bentuk representasi.

Kedua hipotesis dipisahkan dgn membandingkan DAYA PREDIKSI RATING dari
kanal sentimen SAJA (tanpa DeepMF/CBF/fusi), memakai regressor yang sama
(Ridge) pada split yang sama -- hanya REPRESENTASI yang berubah:

  1. keyword_concat_conf  : representasi A2-IRM (taksonomi tetap 4-6 aspek
                            x [skor, confidence]) -- skor dari SA-BERT
                            tersupervisi.
  2. sabert_global_1dim   : SATU skalar SA-BERT global. Kalau ini saja sudah
                            mengalahkan seluruh vektor PyABSA 9-dim, maka
                            masalahnya JELAS bukan struktur/dimensi.
  3. pyabsa_rich9         : order-statistics 9-dim -- persis yang dipakai
                            sel C & D0.
  4. pyabsa_bag_topK      : top-K istilah aspek sbg KOLOM TETAP,
                            [P_pos bila muncul, indikator kehadiran].
                            *** INI BATAS ATAS (UPPER BOUND) UNTUK SETIAP
                            SKEMA CLUSTERING ***: clustering aspek ke
                            taksonomi tetap = penggabungan LINIER kolom-kolom
                            ini, dan Ridge atas himpunan penuh dapat
                            menyatakan penggabungan semacam itu. Jadi bila
                            representasi ini pun masih kalah dari (1), maka
                            TIDAK ADA skema clustering yang bisa menutup gap
                            -- H-struktur tertolak.

Catatan validitas: SA-BERT dilatih HANYA pada train+val (run_baseline.py:113),
jadi TIDAK ada kebocoran test -- A2-IRM tetap sah. Yang ditunjukkan di sini
bukan kebocoran, melainkan bahwa perbandingan "keyword-ABSA vs PyABSA"
TERKONFOUND oleh supervisi domain, sehingga tidak bisa dibaca sebagai
perbandingan metode ABSA an sich.

Usage:
    venv/Scripts/python.exe scripts/diagnose_sentiment_signal_quality.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.pyabsa_scorer import (  # noqa: E402
    load_cached_scores,
    vectorize_absa_features_rich,
)

RATING_COL = "stars"
TOP_K_ASPECTS = 200
RIDGE_ALPHA = 1.0

DOMAINS = {
    "restaurant": dict(split="yelp_restaurant", ckpt="yelp_restaurant", pyabsa="restaurant"),
    "amazon_electronics": dict(
        split="amazon_electronics", ckpt="amazon_electronics", pyabsa="amazon_electronics"
    ),
    "tripadvisor_hotel": dict(
        split="tripadvisor_hotel", ckpt="tripadvisor_hotel", pyabsa="tripadvisor_hotel"
    ),
}
DOMAIN_LABELS = {
    "restaurant": "Restaurant",
    "amazon_electronics": "E-commerce",
    "tripadvisor_hotel": "Hotel",
}


def _load_splits(split_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _REPO_ROOT / "data" / "splits" / split_dir
    cols = ["review_id", RATING_COL]
    train = pd.read_csv(base / "train.csv", usecols=cols)
    test = pd.read_csv(base / "test.csv", usecols=cols)
    return train, test


def _keyword_features(ckpt: str) -> pd.DataFrame:
    path = (
        _REPO_ROOT / "checkpoints" / ckpt / "sentiment_bert" / "absa_concat_confidence_scores.csv"
    )
    return pd.read_csv(path)


def _sabert_global(ckpt: str) -> pd.DataFrame:
    path = _REPO_ROOT / "checkpoints" / ckpt / "sentiment_bert" / "sentiment_scores.csv"
    return pd.read_csv(path)


def _pyabsa_scored(ckpt: str, name: str) -> tuple[pd.DataFrame, dict]:
    pdir = _REPO_ROOT / "checkpoints" / ckpt / "pyabsa"
    scored = load_cached_scores(str(pdir / f"pyabsa_scores_{name}.csv"))
    fb_path = pdir / f"sa_fallback_scores_{name}.csv"
    fallback = {}
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        fallback = dict(zip(fb["review_id"], fb["fallback_score"]))
    return scored, fallback


def _pyabsa_meanpos(scored: pd.DataFrame, fallback: dict) -> np.ndarray:
    """SATU skalar: rata-rata P(pos) antar aspek (fallback bila 0 aspek).
    Strukturnya IDENTIK dgn sabert_global_1dim -- sama-sama satu angka per
    review -- sehingga selisih keduanya mengisolasi KUALITAS SCORER murni,
    dgn struktur representasi dikonstankan sepenuhnya."""
    out = np.zeros((len(scored), 1), dtype=np.float32)
    for r, row in enumerate(scored.itertuples(index=False)):
        if row.n_aspects == 0 or not row.probs:
            out[r, 0] = fallback.get(row.review_id, 0.5)
        else:
            out[r, 0] = float(np.mean([p[2] for p in row.probs]))
    return out


def _cluster_aspect_terms(
    scored: pd.DataFrame, train_ids: set, vocab: list[str], n_clusters: int, seed: int = 42
) -> dict:
    """Induksi KATEGORI ASPEK gaya LSA: matriks kejadian istilah x review
    (HANYA review train) -> TruncatedSVD -> KMeans. Istilah yang muncul di
    review serupa dikelompokkan bersama. Ini realisasi konkret dari
    'PyABSA + clustering ke taksonomi tetap'."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import csr_matrix

    idx = {t: i for i, t in enumerate(vocab)}
    rows, cols = [], []
    for r, row in enumerate(scored.itertuples(index=False)):
        if row.review_id not in train_ids:
            continue
        for term in row.aspects:
            j = idx.get(str(term).lower().strip())
            if j is not None:
                rows.append(r)
                cols.append(j)
    inc = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(scored), len(vocab)),
    )
    # istilah sbg baris: profil distribusional lintas review train
    term_doc = TfidfTransformer().fit_transform(inc).T
    n_comp = min(50, len(vocab) - 1, term_doc.shape[1] - 1)
    emb = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(term_doc)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(emb)
    return {vocab[i]: int(labels[i]) for i in range(len(vocab))}


def _clustered_features(scored: pd.DataFrame, term2cluster: dict, k: int) -> np.ndarray:
    """Format MENIRU keyword concat+confidence: per kategori hasil clustering
    -> [rata-rata P(pos) aspek di kategori itu, fraksi kehadiran]. Posisi
    kolom TETAP lintas review, persis keunggulan struktural taksonomi tetap."""
    out = np.zeros((len(scored), 2 * k), dtype=np.float32)
    for r, row in enumerate(scored.itertuples(index=False)):
        acc: dict[int, list[float]] = {}
        for term, prob in zip(row.aspects, row.probs):
            c = term2cluster.get(str(term).lower().strip())
            if c is not None:
                acc.setdefault(c, []).append(prob[2])
        for c, vals in acc.items():
            out[r, c] = float(np.mean(vals))
            out[r, k + c] = 1.0
    return out


def _bag_of_aspects(scored: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    """Top-K istilah aspek sbg kolom TETAP: [P_pos bila muncul, indikator].
    Identitas aspek dipertahankan PENUH pada posisi tetap -- batas atas untuk
    setiap skema clustering (clustering = penggabungan linier kolom ini)."""
    idx = {t: i for i, t in enumerate(vocab)}
    k = len(vocab)
    out = np.zeros((len(scored), 2 * k), dtype=np.float32)
    for r, row in enumerate(scored.itertuples(index=False)):
        for term, prob in zip(row.aspects, row.probs):
            j = idx.get(str(term).lower().strip())
            if j is None:
                continue
            out[r, j] = prob[2]  # P(pos)
            out[r, k + j] = 1.0  # kehadiran
    return out


def _fit_eval(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> dict:
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    r = float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"rmse": rmse, "pearson_r": r, "n_features": x_train.shape[1]}


def main() -> None:
    rows = []

    for domain, cfg in DOMAINS.items():
        print(f"\n{'=' * 70}\n{DOMAIN_LABELS[domain]}\n{'=' * 70}")
        train, test = _load_splits(cfg["split"])

        kw = _keyword_features(cfg["ckpt"])
        glob = _sabert_global(cfg["ckpt"])
        scored, fallback = _pyabsa_scored(cfg["ckpt"], cfg["pyabsa"])

        # Kosakata aspek dibangun HANYA dari review train -> tidak ada
        # informasi test yang masuk ke definisi fitur.
        train_ids = set(train["review_id"])
        counter: Counter = Counter()
        for row in scored.itertuples(index=False):
            if row.review_id in train_ids:
                for a in row.aspects:
                    t = str(a).lower().strip()
                    if t:
                        counter[t] += 1
        vocab = [t for t, _ in counter.most_common(TOP_K_ASPECTS)]

        rich = pd.DataFrame(
            vectorize_absa_features_rich(scored, fallback_scores=fallback),
            columns=[f"rich_{i}" for i in range(9)],
        )
        rich.insert(0, "review_id", scored["review_id"].values)

        bag = pd.DataFrame(
            _bag_of_aspects(scored, vocab),
            columns=[f"bag_{i}" for i in range(2 * len(vocab))],
        )
        bag.insert(0, "review_id", scored["review_id"].values)

        meanpos = pd.DataFrame(
            _pyabsa_meanpos(scored, fallback), columns=["pyabsa_meanpos"]
        )
        meanpos.insert(0, "review_id", scored["review_id"].values)

        # K = jumlah aspek taksonomi keyword domain ini -> apple-to-apple
        n_kw_aspects = (len(kw.columns) - 1) // 2
        term2cluster = _cluster_aspect_terms(scored, train_ids, vocab, n_kw_aspects)
        clustered = pd.DataFrame(
            _clustered_features(scored, term2cluster, n_kw_aspects),
            columns=[f"clu_{i}" for i in range(2 * n_kw_aspects)],
        )
        clustered.insert(0, "review_id", scored["review_id"].values)

        reps = {
            "1. keyword_concat_conf (A2-IRM, SA-BERT tersupervisi)": kw,
            "2. sabert_global_1dim (satu skalar tersupervisi)": glob,
            "2b. pyabsa_meanpos_1dim (satu skalar, TANPA supervisi)": meanpos,
            "3. pyabsa_rich9 (sel C & D0)": rich,
            "3b. pyabsa_clustered (taksonomi terinduksi, K=|keyword|)": clustered,
            f"4. pyabsa_bag_top{len(vocab)} (BATAS ATAS clustering)": bag,
        }

        for label, feat in reps.items():
            tr = train.merge(feat, on="review_id", how="inner")
            te = test.merge(feat, on="review_id", how="inner")
            fcols = [c for c in feat.columns if c != "review_id"]
            if tr.empty or te.empty:
                print(f"  !! {label}: gagal join (train={len(tr)}, test={len(te)}) -- DILEWATI")
                continue
            res = _fit_eval(
                tr[fcols].to_numpy(np.float64),
                tr[RATING_COL].to_numpy(np.float64),
                te[fcols].to_numpy(np.float64),
                te[RATING_COL].to_numpy(np.float64),
            )
            rows.append(
                {
                    "domain": DOMAIN_LABELS[domain],
                    "representation": label,
                    "n_train": len(tr),
                    "n_test": len(te),
                    **res,
                }
            )
            print(
                f"  {label:<58} dim={res['n_features']:>4}  "
                f"RMSE={res['rmse']:.4f}  r={res['pearson_r']:.4f}"
            )

    out = pd.DataFrame(rows)
    dest = _REPO_ROOT / "reports" / "sentiment_signal_quality.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    print(f"\n\n{'=' * 70}\nRINGKASAN: RMSE prediksi rating dari KANAL SENTIMEN SAJA\n{'=' * 70}")
    print(out.pivot_table(index="representation", columns="domain", values="rmse").to_string())
    print(f"\nDisimpan ke {dest}")
    print(
        "\nCARA MEMBACA:\n"
        "  (2) vs (2b) -> UJI TERBERSIH: struktur identik (sama-sama SATU skalar),\n"
        "                 hanya scorer yang berbeda. Selisihnya = efek supervisi\n"
        "                 domain murni, bebas dari pengaruh bentuk representasi.\n"
        "  (2) < (3)   -> satu skalar tersupervisi mengalahkan 9-dim PyABSA:\n"
        "                 defisit BUKAN soal dimensi/struktur.\n"
        "  (3b) vs (1) -> clustering yang BENAR-BENAR DIJALANKAN (bukan diargumentasikan)\n"
        "                 pada K yang sama dgn taksonomi keyword: inilah jawaban\n"
        "                 langsung atas 'apakah PyABSA+clustering lebih baik?'\n"
        "  (4)         -> identitas aspek penuh pada posisi tetap. Clustering adalah\n"
        "                 penggabungan LINIER kolom-kolom ini, jadi (4) membatasi\n"
        "                 INFORMASI yang tersedia bagi skema clustering mana pun\n"
        "                 (walau clustering bisa unggul dlm hal VARIANS estimasi)."
    )


if __name__ == "__main__":
    main()
