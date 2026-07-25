"""
scripts/measure_cbf_tfidf_leakage.py

Ukur besarnya efek leakage-dalam-train pada fitur TF-IDF CBF: item i punya
`description_text` yang diagregasi dari SELURUH review train item itu
(`build_item_dataframe()`, cbf_clustering.py) -- TERMASUK review milik baris
train (u,i) yang SEDANG dievaluasi. Saat fusion di-training pada baris itu,
`cbf_preds`-nya diturunkan dari profil TF-IDF item yang sebagian dibentuk
oleh review (u,i) itu sendiri.

Metodologi (konsisten dgn test train-side LOO utk profil aspek di
scope_coverage.md/test_scope_guard.py, branch phase2-a2-fusionrs): untuk
sample baris train pada item ber-review-count RENDAH (efek maksimal
terlihat -- 1 review yg dikeluarkan/dimasukkan proporsinya besar), bandingkan
vektor TF-IDF item NAIF (skema produksi, termasuk review target) vs vektor
LEAVE-ONE-OUT (review target dikecualikan), pakai TfidfVectorizer yang SAMA
(fit sekali di korpus naif -- persis skema produksi ItemFeatureBuilder --
HANYA di-transform ulang utk versi LOO, bukan di-fit ulang, supaya
perbandingan terisolasi ke "berubah tidaknya profil 1 item", bukan tercampur
efek pergeseran vocabulary korpus).

Metrik: cosine similarity antara vektor naif vs LOO per baris sampel,
distribusi per-domain, distratifikasi berdasarkan review_count item.

Usage:
    python scripts/measure_cbf_tfidf_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing import TextPreprocessor  # noqa: E402

SEED = 42
RID_COL, USER_COL, ITEM_COL = "review_id", "user_id", "business_id"
REPORT_PATH = _REPO_ROOT / "reports" / "cbf_tfidf_leakage_measurement.md"
N_SAMPLE_PER_DOMAIN = 500
REVIEW_COUNT_BUCKETS = [(1, 5), (6, 15), (16, 50), (51, 10_000)]

DOMAINS = [
    {"label": "amazon_electronics", "split_dir": "data/splits/amazon_electronics"},
    {"label": "restaurant", "split_dir": "data/splits/yelp_restaurant"},
    {"label": "tripadvisor_hotel", "split_dir": "data/splits/tripadvisor_hotel"},
]


def _bucket(n: int) -> str:
    for lo, hi in REVIEW_COUNT_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "?"


def measure_domain(domain: dict) -> dict:
    train = pd.read_csv(Path(domain["split_dir"]) / "train.csv")
    preproc = TextPreprocessor()
    train["text_tfidf"] = train["text"].apply(preproc.clean_for_tfidf)

    # ---- Bangun description_text NAIF per item (skema produksi persis) ----
    item_text_naive = train.groupby(ITEM_COL)["text_tfidf"].apply(lambda x: " ".join(x))
    review_count = train.groupby(ITEM_COL)[RID_COL].count()

    vectorizer = TfidfVectorizer(max_features=500)
    vectorizer.fit(item_text_naive.values)  # vocabulary/IDF TETAP -- persis skema produksi

    # Index item -> daftar (review_id, text_tfidf) -- utk membangun versi LOO
    # dgn FILTER review_id (bukan string-replace, yg rapuh kalau ada >=2
    # review dgn teks identik setelah cleaning -- replace(count=1) bisa
    # membuang kemunculan yg SALAH kalau ada duplikat).
    item_reviews: dict = {
        iid: list(zip(grp[RID_COL], grp["text_tfidf"]))
        for iid, grp in train.groupby(ITEM_COL)
    }

    # ---- Sample baris train, bobot lebih ke item ber-review-count rendah
    # (di situ efek LOO paling kentara) tapi tetap sertakan seluruh rentang
    # utk laporan yang representatif, bukan cuma kasus ekstrem. ----
    train_sample = train.sample(n=min(N_SAMPLE_PER_DOMAIN, len(train)), random_state=SEED)

    rows = []
    for r in train_sample.itertuples(index=False):
        iid = getattr(r, ITEM_COL)
        rid = getattr(r, RID_COL)
        n_rev = int(review_count[iid])

        naive_text = item_text_naive[iid]
        naive_vec = vectorizer.transform([naive_text])

        # Filter TEPAT berdasarkan review_id -- tidak rapuh thd teks duplikat.
        loo_text = " ".join(t for rv, t in item_reviews[iid] if rv != rid).strip()

        loo_vec = vectorizer.transform([loo_text])
        sim = float(cosine_similarity(naive_vec, loo_vec)[0, 0])

        rows.append({
            "review_count": n_rev,
            "bucket": _bucket(n_rev),
            "cosine_sim": sim,
            "shift": 1.0 - sim,
        })

    df = pd.DataFrame(rows)

    bucket_order = [f"{lo}-{hi}" if hi < 10_000 else f"{lo}+" for lo, hi in REVIEW_COUNT_BUCKETS]
    bucket_stats = (
        df.groupby("bucket")["shift"]
        .agg(["count", "mean", "median", lambda s: s.quantile(0.9)])
        .rename(columns={"<lambda_0>": "p90"})
        .reindex(bucket_order)
    )

    return {
        "label": domain["label"],
        "n_train": len(train),
        "n_items": len(item_text_naive),
        "n_sampled": len(df),
        "overall_mean_shift": float(df["shift"].mean()),
        "overall_median_shift": float(df["shift"].median()),
        "overall_p90_shift": float(df["shift"].quantile(0.9)),
        "pct_rows_exposed_low_count": 100.0 * (df["review_count"] <= 15).mean(),
        "bucket_stats": bucket_stats,
    }


def render_domain_block(r: dict) -> str:
    bucket_rows = "\n".join(
        f"| {bucket} | {int(row['count']) if pd.notna(row['count']) else 0} | "
        f"{row['mean']:.4f} | {row['median']:.4f} | {row['p90']:.4f} |"
        for bucket, row in r["bucket_stats"].iterrows()
    )
    return f"""### Domain: `{r['label']}`

n_train={r['n_train']}, n_item unik={r['n_items']}, baris disampel={r['n_sampled']}
(dari total train, `random_state={SEED}`).

**Shift keseluruhan** (1 - cosine similarity antara profil TF-IDF item NAIF
vs LEAVE-ONE-OUT, dirata-rata semua baris sampel): mean={r['overall_mean_shift']:.4f},
median={r['overall_median_shift']:.4f}, p90={r['overall_p90_shift']:.4f}.

{r['pct_rows_exposed_low_count']:.1f}% baris sampel berasal dari item dengan
review_count <=15 (rentang di mana efek 1 review paling terasa).

**Shift per rentang review_count item:**

| review_count | n baris sampel | mean shift | median shift | p90 shift |
|---|---:|---:|---:|---:|
{bucket_rows}
"""


def main() -> None:
    results = [measure_domain(d) for d in DOMAINS]
    blocks_text = "\n".join(render_domain_block(r) for r in results)

    summary_rows = "\n".join(
        f"| `{r['label']}` | {r['overall_mean_shift']:.4f} | {r['overall_median_shift']:.4f} | "
        f"{r['pct_rows_exposed_low_count']:.1f}% |"
        for r in results
    )

    report = f"""# Pengukuran Leakage TF-IDF CBF (leave-one-out-dalam-train)

> Dihasilkan oleh `scripts/measure_cbf_tfidf_leakage.py`. Mengukur seberapa
> besar profil TF-IDF item (`description_text`, `src/baseline/cbf_clustering.py::
> build_item_dataframe`) berubah kalau review milik baris train yang sedang
> dievaluasi DIKECUALIKAN dari agregat item itu sendiri -- proxy langsung
> utk seberapa besar CBF "mengintip" review targetnya sendiri saat training.
> Metodologi: lihat docstring modul. Basis: TRAIN saja, {N_SAMPLE_PER_DOMAIN}
> baris sampel/domain.

## Hasil per domain

{blocks_text}

## Ringkasan lintas domain

| Domain | Mean shift (1-cosine) | Median shift | % baris dari item ber-review-count <=15 |
|---|---:|---:|---:|
{summary_rows}

## Interpretasi

Shift mendekati 0 = profil TF-IDF item PRAKTIS TIDAK BERUBAH kalau review
target dikecualikan (leakage yang terukur dapat diabaikan). Shift mendekati
1 = profil berubah drastis (leakage besar). Bandingkan dgn strata
review_count: kalau shift terkonsentrasi HANYA di item ber-review-count
rendah (efek dilusi -- 1 dari sedikit review = proporsi besar) dan p90/mean
keseluruhan tetap kecil, itu artinya sebagian besar baris (item populer)
PRAKTIS aman, dan risiko nyata terbatas pada ekor sparse (item baru/jarang
direview) -- bukan masalah sistemik di seluruh dataset.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nLaporan ditulis ke {REPORT_PATH}")


if __name__ == "__main__":
    main()
