"""
tests/test_scope_guard.py

Verifikasi adversarial atas infrastruktur Step 3 (ReviewScope + provenance +
guard, `src/data/scope.py` / `src/features/provenance.py` / `src/eval/guards.py`)
SEBELUM Step 4 dimulai. Diminta eksplisit oleh user -- empat pertanyaan yang
belum dijawab test_review_scope.py (Step 3 asli):

1. NEGATIVE CONTROL -- guard benar-benar MENYALA saat leakage terjadi
   (bukan cuma "tidak pernah dites gagal").
2. POSITIVE CONTROL -- provenance toy builder benar-benar terisi di bawah
   legacy scope, bukan diam-diam kosong (yang akan membuat guard lulus
   trivial dan tidak berarti apa-apa).
3. TRAIN-SIDE LOO -- "jebakan implementasi" spec Step 3: profil aspek item
   BERBEDA secara numerik antara versi leave-one-out (HistoricalScope,
   review target dikecualikan) dan versi naif (semua review item, termasuk
   review target baris itu sendiri).
4. COVERAGE -- ukur (bukan asumsikan) seberapa sering review historis
   user/item kosong dan seberapa sering aspek user & item historis SAMA
   SEKALI tidak beririsan, di bawah HistoricalScope. Ditulis ke
   `reports/scope_coverage.md`.

Prinsip: split ASLI (`data/splits/`, Invarian #3, tidak diregenerate) dan
cache ABSA yang SUDAH ada (`absa_aspect_scores.csv`) dipakai apa adanya.
Test 4 memanggil ulang `KeywordAspectSentimentScorer._match_aspects`
(src/legacy, READ-ONLY -- Invarian #2, cuma keyword matching, TIDAK ada
inferensi BERT) karena `absa_aspect_scores.csv` menyimpan SKOR sentimen
per-aspek, bukan indikator "aspek X disebut atau tidak di review ini" --
info yang justru dibutuhkan test 4.

Feature builder yang dipakai test 1-3 sengaja "toy" (identik gaya
scripts/demo_review_scope.py) -- refactor feature builder PRODUKSI
(CBF/ABSA/PyABSA) ke ReviewScope adalah Step 4, bukan cakupan file ini.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.scope import HistoricalScope, TargetVisibleScope  # noqa: E402
from src.eval.guards import TargetLeakageError, assert_no_target_leakage  # noqa: E402
from src.features.provenance import ProvenanceTracker  # noqa: E402
from src.legacy.a2fusionrs.absa_bert import ABSAConfig, KeywordAspectSentimentScorer  # noqa: E402
from src.preprocessing import TextPreprocessor  # noqa: E402

SEED = 42
ITEM_COL = "business_id"
USER_COL = "user_id"
RID_COL = "review_id"

# (label pelaporan, kunci domain ABSAConfig/DEFAULT_ASPECT_KEYWORDS, folder split,
#  folder checkpoint sentiment_bert)
DOMAINS = [
    {
        "label": "amazon_electronics",
        "absa_key": "amazon_electronics",
        "split_dir": "data/splits/amazon_electronics",
        "checkpoint_dir": "checkpoints/amazon_electronics",
    },
    {
        "label": "restaurant",
        "absa_key": "restaurant",
        "split_dir": "data/splits/yelp_restaurant",
        "checkpoint_dir": "checkpoints/yelp_restaurant",
    },
    {
        "label": "tripadvisor_hotel",
        "absa_key": "tripadvisor_hotel",
        "split_dir": "data/splits/tripadvisor_hotel",
        "checkpoint_dir": "checkpoints/tripadvisor_hotel",
    },
]
DOMAIN_IDS = [d["label"] for d in DOMAINS]

REPORT_PATH = _REPO_ROOT / "reports" / "scope_coverage.md"


# ---------------------------------------------------------------------------
# Helper bersama
# ---------------------------------------------------------------------------


def _load_split(domain: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(domain["split_dir"])
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    return train, test


def _load_aspect_scores(domain: dict) -> tuple[pd.DataFrame, list[str]]:
    path = Path(domain["checkpoint_dir"]) / "sentiment_bert" / "absa_aspect_scores.csv"
    df = pd.read_csv(path)
    aspect_cols = [c for c in df.columns if c != RID_COL]
    return df, aspect_cols


def _toy_provenance(scope, eval_rows: pd.DataFrame, corpus: pd.DataFrame):
    """Feature builder TOY (identik prinsip dgn scripts/demo_review_scope.py):
    kandidat = seluruh review item i di korpus, disaring lewat `scope`. Row_id
    == review_id baris (konvensi proyek, lihat ProvenanceTracker)."""
    item_reviews: dict[str, list[str]] = {
        iid: grp[RID_COL].tolist() for iid, grp in corpus.groupby(ITEM_COL)
    }
    prov = ProvenanceTracker()
    eval_index: dict[str, str] = {}
    for r in eval_rows.itertuples(index=False):
        uid = getattr(r, USER_COL)
        iid = getattr(r, ITEM_COL)
        row_id = getattr(r, RID_COL)
        ts = getattr(r, "date", None)
        cand_ids = item_reviews.get(iid, [])
        visible = scope.filter_visible(cand_ids, uid, iid, ts)
        prov.record(row_id, visible)
        eval_index[row_id] = row_id
    return prov, eval_index


# ---------------------------------------------------------------------------
# Test 1 -- NEGATIVE CONTROL: guard harus MENYALA saat leakage diinjeksi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_negative_control_guard_fires_on_injected_leak(domain):
    _, test = _load_split(domain)
    row = test.sample(n=1, random_state=SEED).iloc[0]
    row_id = row[RID_COL]

    # Baseline: provenance BERSIH (review lain, bukan review target). Guard
    # harus lolos -- sanity check sebelum injeksi, supaya kegagalan test di
    # bawah tidak salah dituduhkan ke bug lain.
    prov = ProvenanceTracker()
    prov.record(row_id, {f"decoy_review_{row_id}_1", f"decoy_review_{row_id}_2"})
    eval_index = {row_id: row_id}
    n_checked = assert_no_target_leakage(prov, eval_index)
    assert n_checked == 1

    # INJEKSI SENGAJA: review target masuk provenance -- simulasi persis bug
    # "feature builder lupa lewat HistoricalScope, memakai review (u,i) itu
    # sendiri". Guard WAJIB raise, bukan lolos diam-diam.
    prov.record(row_id, {row_id})
    with pytest.raises(TargetLeakageError, match="TARGET-REVIEW LEAKAGE"):
        assert_no_target_leakage(prov, eval_index)


# ---------------------------------------------------------------------------
# Test 2 -- POSITIVE CONTROL: provenance benar-benar terisi di bawah legacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_positive_control_provenance_actually_populated(domain):
    train, test = _load_split(domain)
    eval_rows = test.sample(n=min(500, len(test)), random_state=SEED)

    # Korpus WAJIB mencakup baris test yg dievaluasi (bukan cuma train) --
    # supaya review target benar-benar hadir sbg kandidat yg bisa diloloskan
    # TargetVisibleScope. Dipersempit ke item yg relevan demi kecepatan
    # (sama prinsip dgn scripts/demo_review_scope.py).
    items = set(eval_rows[ITEM_COL].unique())
    corpus = pd.concat([train, test], ignore_index=True)
    corpus = corpus[corpus[ITEM_COL].isin(items)].reset_index(drop=True)

    scope = TargetVisibleScope(corpus, item_col=ITEM_COL, user_col=USER_COL)
    prov, eval_index = _toy_provenance(scope, eval_rows, corpus)

    n_rows = len(eval_rows)
    n_contains_target = sum(1 for rid, trid in eval_index.items() if trid in prov.get(rid))
    n_nonempty = sum(1 for rid in eval_index if len(prov.get(rid)) > 0)

    assert n_contains_target == n_rows, (
        f"[{domain['label']}] {n_rows - n_contains_target}/{n_rows} baris TIDAK memuat "
        "review targetnya sendiri di provenance. Di bawah TargetVisibleScope (legacy) "
        "ini SEHARUSNYA selalu terjadi -- review target ada di korpus dan scope legacy "
        "tidak mengecualikan apa pun."
    )
    frac_nonempty = n_nonempty / n_rows
    assert frac_nonempty >= 0.99, (
        f"[{domain['label']}] hanya {n_nonempty}/{n_rows} ({frac_nonempty:.1%}) baris "
        "punya provenance TIDAK KOSONG. Kalau feature builder toy diam-diam mengembalikan "
        "set kosong, guard di Step 3 lulus trivial dan tidak menguji apa pun -- ini "
        "positive control yang membuktikan itu TIDAK terjadi di sini."
    )


# ---------------------------------------------------------------------------
# Test 3 -- TRAIN-SIDE LOO: jebakan implementasi spec Step 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS, ids=DOMAIN_IDS)
def test_train_side_leave_one_out_changes_aspect_profile(domain):
    train, _test = _load_split(domain)
    aspect_df, aspect_cols = _load_aspect_scores(domain)
    aspect_by_rid = aspect_df.set_index(RID_COL)[aspect_cols]

    review_count = train.groupby(ITEM_COL)[RID_COL].count()
    candidate_items = review_count[(review_count >= 2) & (review_count <= 5)].index.tolist()
    assert len(candidate_items) > 0, (
        f"[{domain['label']}] tidak ada item dgn review_count train di [2,5] -- "
        "tidak bisa menjalankan test LOO train (longgarkan rentang kalau ini terjadi)."
    )
    rng = np.random.RandomState(SEED)
    n_pick = min(50, len(candidate_items))
    chosen_items = rng.choice(np.array(candidate_items, dtype=object), size=n_pick, replace=False)

    scope = HistoricalScope(train, item_col=ITEM_COL, user_col=USER_COL)

    n_checked = 0
    n_differ = 0
    identical_examples: list[str] = []

    for iid in chosen_items:
        item_rows = train[train[ITEM_COL] == iid]
        item_rids = item_rows[RID_COL].tolist()
        all_vectors = aspect_by_rid.reindex(item_rids).dropna()
        if all_vectors.empty:
            continue
        # Profil NAIF: mean per-aspek atas SEMUA review item, TERMASUK review
        # target baris yg sedang diperiksa -- ini versi "salah" yg harus
        # dikalahkan HistoricalScope.
        profile_naive = all_vectors.mean(axis=0).to_numpy()

        for r in item_rows.itertuples(index=False):
            uid = getattr(r, USER_COL)
            row_rid = getattr(r, RID_COL)
            ts = getattr(r, "date", None)
            visible = scope.filter_visible(item_rids, uid, iid, ts)
            visible_rids = [rid for rid in item_rids if rid in visible]
            if not visible_rids:
                continue  # item cuma py review target ini sendiri -- tidak ada basis LOO
            scoped_vectors = aspect_by_rid.reindex(visible_rids).dropna()
            if scoped_vectors.empty:
                continue
            # Profil LOO: mean per-aspek HANYA atas review yg visible di bawah
            # HistoricalScope -- review target (u,i) sudah tercabut.
            profile_scoped = scoped_vectors.mean(axis=0).to_numpy()

            n_checked += 1
            # rtol=0 SENGAJA: skor beberapa item mengelompok sangat rapat dekat
            # 1.0 (fallback whole-review), jadi rtol default (1e-5) numpy
            # menoleransi selisih ~1e-7 yang justru BUKTI LOO bekerja (lihat
            # investigasi kegagalan pertama test ini) -- atol murni tanpa
            # komponen relatif adalah kriteria yang benar di sini.
            if np.allclose(profile_scoped, profile_naive, atol=1e-9, rtol=0.0):
                identical_examples.append(f"item={iid} review={row_rid}")
            else:
                n_differ += 1

    assert n_checked > 0, (
        f"[{domain['label']}] tidak ada baris train yg bisa diperiksa (semua item "
        "kandidat cuma py 1 review setelah LOO)."
    )
    assert n_differ == n_checked, (
        f"[{domain['label']}] {n_checked - n_differ}/{n_checked} baris train profil "
        "aspeknya IDENTIK antara versi LOO (HistoricalScope) dan versi naif (semua "
        "review item, termasuk review targetnya sendiri) -- LOO di dalam split train "
        "TIDAK terpasang (jebakan implementasi spec Step 3, baris 130-133). Contoh: "
        f"{identical_examples[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 4 -- COVERAGE: ukur, jangan diasumsikan. Tulis reports/scope_coverage.md
# ---------------------------------------------------------------------------


def _mentioned_aspects(text: str, preproc: TextPreprocessor, scorer: KeywordAspectSentimentScorer) -> set[str]:
    """Aspek yg DISEBUT (keyword match, BUKAN skor sentimen) di sebuah review.
    Murni string matching -- tidak ada panggilan model, konsisten dgn
    `KeywordAspectSentimentScorer._match_aspects` yg dipakai run_baseline_absa.py."""
    text_bert = preproc.clean_for_bert(text)
    sentences = scorer._split_sentences(text_bert)
    matches = scorer._match_aspects(sentences)
    return set(matches.keys())


def _coverage_for_domain(domain: dict, n_eval: int = 500) -> dict:
    train, test = _load_split(domain)
    eval_rows = test.sample(n=min(n_eval, len(test)), random_state=SEED)

    train_by_user = {uid: grp for uid, grp in train.groupby(USER_COL)}
    train_by_item = {iid: grp for iid, grp in train.groupby(ITEM_COL)}

    preproc = TextPreprocessor()
    scorer = KeywordAspectSentimentScorer(None, config=ABSAConfig(domain=domain["absa_key"]))

    # Cache per entity (bukan per review) -- item/user populer dipakai ulang
    # antar baris eval tanpa menghitung ulang union aspeknya.
    user_aspect_cache: dict[str, frozenset] = {}
    item_aspect_cache: dict[str, frozenset] = {}

    def user_aspect_set(uid: str) -> frozenset:
        if uid not in user_aspect_cache:
            grp = train_by_user.get(uid)
            if grp is None:
                user_aspect_cache[uid] = frozenset()
            else:
                s: set[str] = set()
                for text in grp["text"]:
                    s |= _mentioned_aspects(text, preproc, scorer)
                user_aspect_cache[uid] = frozenset(s)
        return user_aspect_cache[uid]

    def item_aspect_set(iid: str) -> frozenset:
        if iid not in item_aspect_cache:
            grp = train_by_item.get(iid)
            if grp is None:
                item_aspect_cache[iid] = frozenset()
            else:
                s: set[str] = set()
                for text in grp["text"]:
                    s |= _mentioned_aspects(text, preproc, scorer)
                item_aspect_cache[iid] = frozenset(s)
        return item_aspect_cache[iid]

    n_rows = len(eval_rows)
    n_zero_user_hist = 0
    n_zero_item_hist = 0
    n_zero_overlap = 0

    for r in eval_rows.itertuples(index=False):
        uid = getattr(r, USER_COL)
        iid = getattr(r, ITEM_COL)

        n_user_hist = len(train_by_user.get(uid, train.iloc[0:0]))
        n_item_hist = len(train_by_item.get(iid, train.iloc[0:0]))
        if n_user_hist == 0:
            n_zero_user_hist += 1
        if n_item_hist == 0:
            n_zero_item_hist += 1

        if n_user_hist == 0 or n_item_hist == 0:
            n_zero_overlap += 1  # tidak ada basis apa pun utk overlap
            continue
        if not (user_aspect_set(uid) & item_aspect_set(iid)):
            n_zero_overlap += 1

    return {
        "label": domain["label"],
        "n_eval": n_rows,
        "pct_zero_user_hist": 100.0 * n_zero_user_hist / n_rows,
        "pct_zero_item_hist": 100.0 * n_zero_item_hist / n_rows,
        "pct_zero_aspect_overlap": 100.0 * n_zero_overlap / n_rows,
        "n_unique_users_touched": len(user_aspect_cache),
        "n_unique_items_touched": len(item_aspect_cache),
    }


def test_coverage_report_written_for_all_domains():
    results = [_coverage_for_domain(d) for d in DOMAINS]

    lines = [
        "# Scope Coverage -- Fase 1 Step 3 (pra-Step 4)",
        "",
        "> Dihasilkan oleh `tests/test_scope_guard.py::test_coverage_report_written_for_all_domains`.",
        "> Di bawah `HistoricalScope` (protokol deployment-valid): review historis = review",
        "> TRAIN milik user/item tsb (baris eval diambil dari TEST, jadi review test itu",
        "> sendiri tidak pernah termasuk -- tidak perlu LOO eksplisit di sini, beda dgn Test 3",
        "> yang khusus memeriksa baris TRAIN). 500 baris test/domain (`random_state=42`).",
        "> \"Aspek beririsan\" = irisan set aspek yang DISEBUT (keyword match, bukan skor",
        "> sentimen) di seluruh review historis user U dan seluruh review historis item I;",
        "> baris dgn 0 review historis (user ATAU item) otomatis dihitung sbg 0 aspek",
        "> beririsan (tidak ada basis apa pun).",
        "",
        "| Domain | n eval | % 0 review historis user | % 0 review historis item | % 0 aspek beririsan |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['label']}` | {r['n_eval']} | {r['pct_zero_user_hist']:.1f}% | "
            f"{r['pct_zero_item_hist']:.1f}% | {r['pct_zero_aspect_overlap']:.1f}% |"
        )
    lines += [
        "",
        "Entitas unik yang benar-benar diproses (keyword matching, bukan disimulasikan):",
        "",
        "| Domain | user unik tersentuh | item unik tersentuh |",
        "|---|---:|---:|",
    ]
    for r in results:
        lines.append(f"| `{r['label']}` | {r['n_unique_users_touched']} | {r['n_unique_items_touched']} |")
    lines += [
        "",
        "**Implikasi utk Step 4.** % 0 aspek beririsan tinggi berarti fitur aspek-personal",
        "(irisan preferensi aspek user x profil aspek item) tidak akan punya sinyal untuk",
        "porsi baris eval sebesar itu di bawah protokol deployment-valid -- P2/P3 perlu",
        "fallback (mis. rata-rata global) utk baris-baris tsb, konsisten dgn desain P2 di",
        "phase1_spec.md (\"Untuk baris evaluasi, ganti ... dengan nilai rata-rata train global\").",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Sanity struktural (deskriptif, BUKAN klaim benar/salah soal sparsity itu
    # sendiri -- angka sparsity APA ADANYA yang mau diketahui, bukan diuji lulus/gagal).
    assert REPORT_PATH.exists()
    assert len(results) == len(DOMAINS)
    for r in results:
        assert r["n_eval"] > 0
        for key in ("pct_zero_user_hist", "pct_zero_item_hist", "pct_zero_aspect_overlap"):
            assert 0.0 <= r[key] <= 100.0

    print(f"\nreports/scope_coverage.md ditulis ({len(results)} domain).")
    for r in results:
        print(
            f"  {r['label']}: 0-user-hist={r['pct_zero_user_hist']:.1f}% "
            f"0-item-hist={r['pct_zero_item_hist']:.1f}% "
            f"0-aspect-overlap={r['pct_zero_aspect_overlap']:.1f}%"
        )
