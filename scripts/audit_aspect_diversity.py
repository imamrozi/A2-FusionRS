"""
scripts/audit_aspect_diversity.py

Diagnostik pra-Fase 2 (diminta user): seberapa DIVERSE kosakata aspek
sebenarnya di tiap domain, dipakai sbg kandidat variabel MODERATOR (mis.
"item/domain dgn aspek lebih beragam mendapat manfaat fusion lebih besar").

Sumber data: cache PyABSA (`checkpoints/{domain}/pyabsa/pyabsa_scores_*.csv`,
kolom `aspects_json`) -- BUKAN taksonomi keyword 4-6 kategori tetap yang
dipakai laporan sebelumnya (`aspect_identifiability.md`). PyABSA bersifat
OPEN-VOCABULARY (ekstraksi term bebas, mis. "setup" vs "set up" sbg entri
terpisah) -- itulah alasan "20 aspek terbanyak" & "normalisasi vocab"
bermakna di sini, tidak bermakna utk taksonomi 4-6-kategori.

Normalisasi vocab (RINGAN, didokumentasikan eksplisit -- BUKAN clustering
sinonim penuh): lowercase -> strip -> collapse whitespace -> lemmatisasi
per-kata (WordNetLemmatizer, POS noun) -> gabung ulang. Ini menyatukan
variasi morfologis ("images"->"image", "batteries"->"battery) TAPI TIDAK
menyatukan sinonim/frasa berbeda ("setup" vs "set up" TETAP 2 entri
berbeda) -- keterbatasan yang disengaja, bukan bug, supaya scope tetap
proporsional dgn diagnostik cepat (bukan proyek NLP clustering sendiri).

Basis data: TRAIN saja (konsisten dgn reports/aspect_identifiability.md),
join `pyabsa_scores_*.csv` ke `train.csv` lewat `review_id`.

Usage:
    python scripts/audit_aspect_diversity.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing import ensure_nltk_resources  # noqa: E402

RID_COL = "review_id"
REPORT_PATH = _REPO_ROOT / "reports" / "aspect_diversity.md"
TOP_N = 20

DOMAINS = [
    {"label": "amazon_electronics", "split_dir": "data/splits/amazon_electronics", "pyabsa_file": "checkpoints/amazon_electronics/pyabsa/pyabsa_scores_amazon_electronics.csv"},
    {"label": "restaurant", "split_dir": "data/splits/yelp_restaurant", "pyabsa_file": "checkpoints/yelp_restaurant/pyabsa/pyabsa_scores_restaurant.csv"},
    {"label": "tripadvisor_hotel", "split_dir": "data/splits/tripadvisor_hotel", "pyabsa_file": "checkpoints/tripadvisor_hotel/pyabsa/pyabsa_scores_tripadvisor_hotel.csv"},
]


def _normalize_term(term: str, lemmatizer: WordNetLemmatizer) -> str:
    term = term.strip().lower()
    term = " ".join(term.split())  # collapse whitespace berlebih
    if not term:
        return term
    words = [lemmatizer.lemmatize(w, pos="n") for w in term.split(" ")]
    return " ".join(words)


def _shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient standar atas array hitungan (>=0). 0 = merata
    sempurna antar aspek, mendekati 1 = terkonsentrasi ke sedikit aspek."""
    x = np.sort(counts.astype(float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def audit_domain(domain: dict, lemmatizer: WordNetLemmatizer) -> dict:
    train_rids = set(pd.read_csv(Path(domain["split_dir"]) / "train.csv", usecols=[RID_COL])[RID_COL])
    pyabsa = pd.read_csv(domain["pyabsa_file"])
    pyabsa = pyabsa[pyabsa[RID_COL].isin(train_rids)].reset_index(drop=True)

    per_review_terms: list[list[str]] = []
    global_counts: Counter = Counter()

    for raw in pyabsa["aspects_json"]:
        try:
            terms = json.loads(raw) if isinstance(raw, str) else []
        except (json.JSONDecodeError, TypeError):
            terms = []
        normed = sorted({_normalize_term(t, lemmatizer) for t in terms if isinstance(t, str) and t.strip()})
        per_review_terms.append(normed)
        global_counts.update(normed)

    n_aspects_per_review = np.array([len(t) for t in per_review_terms])
    counts_arr = np.array(list(global_counts.values()))
    total_mentions = int(counts_arr.sum())

    ranked = global_counts.most_common(TOP_N)

    return {
        "label": domain["label"],
        "n_train_reviews_matched": len(pyabsa),
        "n_unique_aspects": len(global_counts),
        "total_mentions": total_mentions,
        "entropy_bits": _shannon_entropy(counts_arr) if total_mentions else float("nan"),
        "entropy_max_bits": float(np.log2(len(global_counts))) if len(global_counts) else float("nan"),
        "gini": _gini(counts_arr),
        "mean_aspects_per_review": float(n_aspects_per_review.mean()) if len(n_aspects_per_review) else float("nan"),
        "median_aspects_per_review": float(np.median(n_aspects_per_review)) if len(n_aspects_per_review) else float("nan"),
        "top_n": [(term, cnt, 100.0 * cnt / total_mentions) for term, cnt in ranked],
    }


def render_domain_block(r: dict) -> str:
    top_lines = "\n".join(
        f"| {i+1} | {term} | {cnt} | {share:.2f}% |"
        for i, (term, cnt, share) in enumerate(r["top_n"])
    )
    entropy_pct_of_max = (
        100.0 * r["entropy_bits"] / r["entropy_max_bits"] if r["entropy_max_bits"] else float("nan")
    )
    return f"""### Domain: `{r['label']}`

Review train tercakup cache PyABSA: {r['n_train_reviews_matched']}. Total mention
aspek (setelah normalisasi, sebelum dedup lintas-review): {r['total_mentions']}.

| Metrik | Nilai |
|---|---:|
| Jumlah aspek unik (setelah normalisasi vocab) | {r['n_unique_aspects']} |
| Entropi Shannon distribusi aspek | {r['entropy_bits']:.2f} bit (maks teoritis {r['entropy_max_bits']:.2f} bit = {entropy_pct_of_max:.1f}% dari maks) |
| Gini coefficient | {r['gini']:.3f} |
| Rata-rata aspek unik per review | {r['mean_aspects_per_review']:.2f} |
| Median aspek unik per review | {r['median_aspects_per_review']:.1f} |

**Top {TOP_N} aspek terbanyak:**

| # | Aspek (dinormalisasi) | Mention | % dari total |
|---:|---|---:|---:|
{top_lines}
"""


def main() -> None:
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()

    results = [audit_domain(d, lemmatizer) for d in DOMAINS]
    blocks_text = "\n".join(render_domain_block(r) for r in results)

    summary_rows = "\n".join(
        f"| `{r['label']}` | {r['n_unique_aspects']} | {r['entropy_bits']:.2f} | {r['gini']:.3f} | "
        f"{r['mean_aspects_per_review']:.2f} |"
        for r in results
    )

    report = f"""# Aspect Diversity -- Diagnostik Pra-Fase 2 (kandidat variabel moderator)

> Dihasilkan oleh `scripts/audit_aspect_diversity.py`. Sumber: cache PyABSA
> open-vocabulary (`aspects_json`), BUKAN taksonomi keyword 4-6-kategori
> tetap yang dipakai `aspect_identifiability.md`. Basis: TRAIN saja.
> Normalisasi vocab RINGAN (case/whitespace/lemma), BUKAN clustering
> sinonim penuh -- lihat docstring modul untuk detail & keterbatasan.

## Hasil per domain

{blocks_text}

## Ringkasan lintas domain

| Domain | Aspek unik | Entropi (bit) | Gini | Rata-rata aspek/review |
|---|---:|---:|---:|---:|
{summary_rows}

**Catatan pembacaan.** Entropi tinggi + Gini rendah = kosakata aspek dipakai
merata (kandidat baik utk moderator "domain aspek-beragam"). Gini tinggi
berarti segelintir aspek mendominasi mention walau kosakata NOMINAL luas --
lebih dekat ke taksonomi 4-6-kategori `aspect_identifiability.md` secara
efektif, meski secara nominal kosakata PyABSA jauh lebih besar.

**Catatan kualitas data (diobservasi, tidak difilter diam-diam).** Beberapa
entri top-20 adalah noise ekstraksi ATEPC, bukan aspek sungguhan -- mis.
`they`/`it` (restaurant, hotel) dan simbol `$` (restaurant). Angka
laporan ini APA ADANYA dari cache PyABSA (tidak ada post-filtering manual)
supaya tidak menyembunyikan karakteristik nyata pipeline ekstraksi; kalau
metrik ini dipakai lebih jauh (mis. jadi fitur moderator), pembersihan
stopword/pronoun eksplisit perlu ditambahkan sbg langkah terpisah yang
didokumentasikan.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nLaporan ditulis ke {REPORT_PATH}")


if __name__ == "__main__":
    main()
