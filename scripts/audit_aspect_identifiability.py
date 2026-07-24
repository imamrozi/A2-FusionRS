"""
scripts/audit_aspect_identifiability.py

Diagnostik pra-Fase 2 (diminta user, bukan bagian literal docs/phase1_spec.md):
apakah data TRAIN yang ada punya cukup sinyal untuk mengidentifikasi bobot
preferensi aspek per-user w_u (ide "aspect-bridge" -- decomposisi importance x
quality) SEBELUM waktu dihabiskan membangunnya. Kalau mention aspek per user
jarang/terkonsentrasi ke sedikit aspek/tidak stabil antar-subset review, w_u
tidak akan teridentifikasi dengan baik walau arsitekturnya benar.

Definisi "mention" (dipakai konsisten di seluruh laporan): SATU mention =
SATU pasangan (review, aspek) di mana aspek itu ke-match minimal 1 kalimat
di review itu (keyword matching `KeywordAspectSentimentScorer._match_aspects`,
src/legacy, READ-ONLY -- Invarian #2, TIDAK ada inferensi BERT). Multiplisitas
kalimat DALAM satu review utk aspek yg sama TIDAK dihitung berlipat -- 1 review
x 1 aspek = maksimal 1 mention, konsisten dgn semantik "user menyebut aspek X
di review ini", bukan "seberapa banyak kalimat".

Section 6 (split-half reliability) memakai profil FREKUENSI mention per aspek
(vektor hitungan mention per aspek dalam satu belahan random review user),
BUKAN skor sentimen -- karena yang diuji adalah stabilitas SALIENCE/preferensi
("aspek mana yang lebih sering dibicarakan user ini"), bukan opini/kualitas.
Pearson correlation tidak berubah oleh normalisasi skala per-vektor, jadi
hitungan mentah (bukan proporsi) sudah cukup.

Usage:
    python scripts/audit_aspect_identifiability.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.legacy.a2fusionrs.absa_bert import ABSAConfig, KeywordAspectSentimentScorer  # noqa: E402
from src.preprocessing import TextPreprocessor  # noqa: E402

SEED = 42
USER_COL = "user_id"
ITEM_COL = "business_id"
RID_COL = "review_id"
REPORT_PATH = _REPO_ROOT / "reports" / "aspect_identifiability.md"
PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

DOMAINS = [
    {"label": "amazon_electronics", "absa_key": "amazon_electronics", "split_dir": "data/splits/amazon_electronics"},
    {"label": "restaurant", "absa_key": "restaurant", "split_dir": "data/splits/yelp_restaurant"},
    {"label": "tripadvisor_hotel", "absa_key": "tripadvisor_hotel", "split_dir": "data/splits/tripadvisor_hotel"},
]


def _compute_mentioned_aspects(train: pd.DataFrame, scorer: KeywordAspectSentimentScorer, preproc: TextPreprocessor) -> list[frozenset]:
    """1 kali keyword-matching per review train, urutan sejajar dgn `train`."""
    out = []
    for text in train["text"]:
        text_bert = preproc.clean_for_bert(text)
        sentences = scorer._split_sentences(text_bert)
        matches = scorer._match_aspects(sentences)
        out.append(frozenset(matches.keys()))
    return out


def _entity_mention_stats(groups: dict, aspect_names: list[str]) -> tuple[list[int], list[int], dict[str, Counter]]:
    """Untuk tiap entitas (user atau item): total mention (jumlah pasangan
    (review,aspek)), jumlah aspek unik, dan Counter mention per aspek."""
    totals, uniques, counts_by_entity = [], [], {}
    for eid, aspect_sets in groups.items():
        counts = Counter()
        for s in aspect_sets:
            counts.update(s)
        totals.append(sum(counts.values()))
        uniques.append(len(counts))
        counts_by_entity[eid] = counts
    return totals, uniques, counts_by_entity


def _percentile_row(values: list[int]) -> dict:
    arr = np.array(values, dtype=float)
    return {f"p{int(q*100)}": float(np.percentile(arr, q * 100)) for q in PERCENTILES}


def _split_half_reliability(user_aspect_sets: dict, aspect_names: list[str], min_reviews: int = 4):
    rng = np.random.RandomState(SEED)
    correlations = []
    n_skipped_too_few = 0
    n_skipped_degenerate = 0

    for uid in sorted(user_aspect_sets.keys()):
        aspect_sets = user_aspect_sets[uid]
        n = len(aspect_sets)
        if n < min_reviews:
            n_skipped_too_few += 1
            continue
        idx = np.arange(n)
        rng.shuffle(idx)
        half = n // 2
        idx_a, idx_b = idx[:half], idx[half:]

        counts_a, counts_b = Counter(), Counter()
        for i in idx_a:
            counts_a.update(aspect_sets[i])
        for i in idx_b:
            counts_b.update(aspect_sets[i])

        vec_a = np.array([counts_a.get(a, 0) for a in aspect_names], dtype=float)
        vec_b = np.array([counts_b.get(a, 0) for a in aspect_names], dtype=float)
        if vec_a.std() == 0 or vec_b.std() == 0:
            n_skipped_degenerate += 1
            continue
        r, _ = pearsonr(vec_a, vec_b)
        correlations.append(r)

    return correlations, n_skipped_too_few, n_skipped_degenerate


def audit_domain(domain: dict) -> dict:
    train = pd.read_csv(Path(domain["split_dir"]) / "train.csv", usecols=[RID_COL, USER_COL, ITEM_COL, "text"])
    preproc = TextPreprocessor()
    scorer = KeywordAspectSentimentScorer(None, config=ABSAConfig(domain=domain["absa_key"]))
    aspect_names = list(scorer.aspect_keywords.keys())

    print(f"[{domain['label']}] keyword-matching {len(train)} review train ...")
    aspect_sets = _compute_mentioned_aspects(train, scorer, preproc)
    train = train.assign(_aspects=aspect_sets)

    user_groups = train.groupby(USER_COL)["_aspects"].apply(list).to_dict()
    item_groups = train.groupby(ITEM_COL)["_aspects"].apply(list).to_dict()

    # ---- Section 1 & 2: per-user ----
    user_totals, user_uniques, user_aspect_counts = _entity_mention_stats(user_groups, aspect_names)

    # ---- Section 3: vocab efektif (agregat SELURUH train, bukan per user) ----
    global_counts = Counter()
    for s in aspect_sets:
        global_counts.update(s)
    total_mentions_global = sum(global_counts.values())
    ranked = sorted(global_counts.items(), key=lambda kv: -kv[1])
    cum = 0
    n_for_80 = n_for_95 = len(ranked)
    for i, (_a, c) in enumerate(ranked, start=1):
        cum += c
        if cum / total_mentions_global >= 0.80 and n_for_80 == len(ranked):
            n_for_80 = i
        if cum / total_mentions_global >= 0.95 and n_for_95 == len(ranked):
            n_for_95 = i
            break

    # ---- Section 4: kunci identifiability kasar ----
    n_qualify = sum(
        1 for counts in user_aspect_counts.values()
        if sum(1 for c in counts.values() if c >= 3) >= 3
    )
    n_users = len(user_aspect_counts)
    pct_qualify = 100.0 * n_qualify / n_users if n_users else 0.0

    # ---- Section 5: sisi item ----
    item_totals, item_uniques, _item_aspect_counts = _entity_mention_stats(item_groups, aspect_names)

    # ---- Section 6: split-half reliability ----
    correlations, n_skip_few, n_skip_degen = _split_half_reliability(user_groups, aspect_names)

    return {
        "label": domain["label"],
        "aspect_names": aspect_names,
        "n_train": len(train),
        "n_users": n_users,
        "n_items": len(item_groups),
        "user_mentions_pct": _percentile_row(user_totals),
        "user_unique_pct": _percentile_row(user_uniques),
        "n_aspects_nominal": len(aspect_names),
        "n_for_80": n_for_80,
        "n_for_95": n_for_95,
        "total_mentions_global": total_mentions_global,
        "aspect_share": [(a, c, 100.0 * c / total_mentions_global) for a, c in ranked],
        "n_qualify": n_qualify,
        "pct_qualify": pct_qualify,
        "item_mentions_pct": _percentile_row(item_totals),
        "item_unique_pct": _percentile_row(item_uniques),
        "n_correlations": len(correlations),
        "mean_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "median_correlation": float(np.median(correlations)) if correlations else float("nan"),
        "n_skip_few_reviews": n_skip_few,
        "n_skip_degenerate": n_skip_degen,
    }


def _pct_table(pct: dict) -> str:
    return " | ".join(f"{pct[f'p{q}']:.2f}" for q in (10, 25, 50, 75, 90))


def render_domain_block(r: dict) -> str:
    aspect_share_lines = "\n".join(
        f"| {a} | {c} | {s:.1f}% |" for a, c, s in r["aspect_share"]
    )
    return f"""### Domain: `{r['label']}`

n_train = {r['n_train']} review, {r['n_users']} user unik, {r['n_items']} item unik.
Taksonomi aspek nominal ({r['n_aspects_nominal']}): {', '.join(r['aspect_names'])}.

**1. Distribusi total mention aspek per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| {_pct_table(r['user_mentions_pct'])} |

**2. Distribusi jumlah aspek UNIK per user (train)**

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| {_pct_table(r['user_unique_pct'])} |

(maksimum mungkin = {r['n_aspects_nominal']}, sama dgn ukuran vocab nominal)

**3. Ukuran kosakata aspek EFEKTIF (bukan nominal)**

Total mention di seluruh train: {r['total_mentions_global']}. Aspek yang menutupi:
- 80% total mention: **{r['n_for_80']}** dari {r['n_aspects_nominal']} aspek nominal
- 95% total mention: **{r['n_for_95']}** dari {r['n_aspects_nominal']} aspek nominal

Distribusi share per aspek (diurut menurun):

| Aspek | Total mention | % dari total |
|---|---:|---:|
{aspect_share_lines}

**4. Kunci: user dgn >=3 mention utk >=3 aspek berbeda**

**{r['pct_qualify']:.1f}%** ({r['n_qualify']}/{r['n_users']}) user train memenuhi ambang
ini -- kandidat kasar "punya harapan w_u teridentifikasi per-user". Sisanya
({100 - r['pct_qualify']:.1f}%) TIDAK cukup data personal utk mengestimasi bobot
{r['n_aspects_nominal']} dimensi aspek secara andal per-user.

**5. Sisi item (pembanding)**

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Total mention per item | {_pct_table(r['item_mentions_pct'])} |
| Aspek unik per item | {_pct_table(r['item_unique_pct'])} |

**6. Split-half reliability (stabilitas profil frekuensi-aspek user)**

User dgn >=4 review train dibagi acak jadi 2 belahan (`seed={SEED}`), profil =
vektor hitungan mention per aspek per belahan, korelasi Pearson antar-belahan
per user, dirata-rata.

- User diperiksa (>=4 review): {r['n_correlations'] + r['n_skip_degenerate']}
- Dilewati (review train <4): {r['n_skip_few_reviews']}
- Dilewati (salah satu belahan varians nol -- korelasi tak terdefinisi): {r['n_skip_degenerate']}
- **Korelasi rata-rata antar-belahan: {r['mean_correlation']:.3f}** (median: {r['median_correlation']:.3f}, n={r['n_correlations']})
"""


def main() -> None:
    results = [audit_domain(domain) for domain in DOMAINS]
    blocks_text = "\n".join(render_domain_block(r) for r in results)

    summary_rows = "\n".join(
        f"| `{r['label']}` | {r['n_for_80']}/{r['n_aspects_nominal']} | {r['n_for_95']}/{r['n_aspects_nominal']} | "
        f"{r['pct_qualify']:.1f}% | {r['mean_correlation']:.3f} (n={r['n_correlations']}) |"
        for r in results
    )

    report = f"""# Aspect Identifiability -- Diagnostik Pra-Fase 2

> Dihasilkan oleh `scripts/audit_aspect_identifiability.py`. Mengukur apakah
> data TRAIN yang ada (k-core sudah difilter, lihat laporan sebelumnya) punya
> cukup sinyal untuk mengidentifikasi preferensi aspek per-user (w_u) --
> SEBELUM waktu diinvestasikan membangun arsitektur yang mengasumsikannya.
> Definisi "mention" dan metodologi tiap section: lihat docstring modul.

## Hasil per domain

{blocks_text}

## Ringkasan lintas domain

| Domain | Vocab efektif @80% | Vocab efektif @95% | % user >=3 mention x >=3 aspek | Korelasi split-half rata-rata |
|---|---:|---:|---:|---:|
{summary_rows}

**Catatan pembacaan.** Section 4 dan 6 adalah dua sudut pandang atas
pertanyaan yang sama: Section 4 mengukur KECUKUPAN data (cross-sectional,
sekali potong) per user; Section 6 mengukur STABILITAS sinyal itu sendiri
(kalau dibagi dua secara acak, apakah "aspek favorit" user tetap konsisten).
%% qualify tinggi tapi korelasi split-half rendah berarti masalahnya BUKAN
kekurangan data mentah, melainkan preferensi aspek itu sendiri TIDAK stabil/
noise-dominated pada level individual -- dua kesimpulan yang punya implikasi
desain berbeda utk Fase 2.
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nLaporan ditulis ke {REPORT_PATH}")


if __name__ == "__main__":
    main()
