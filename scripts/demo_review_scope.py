"""
scripts/demo_review_scope.py

Fase 1 Step 3 -- demonstrasi end-to-end infrastruktur ReviewScope + provenance +
guard, memakai SPLIT ASLI yang sudah ada (Invarian #3: tidak me-regenerate
split). Feature builder PRODUKSI (CBF/ABSA/PyABSA/BERT di src/legacy/) SENGAJA
tidak disentuh -- refactor-nya adalah Step 4. Skrip ini memakai feature builder
"toy" (rata-rata stars dari review item yang terlihat) hanya untuk membuktikan
bahwa jalur ReviewScope -> ProvenanceTracker -> guard bekerja persis seperti
kontrak spec Step 3.

Dibuktikan tiga skenario (memenuhi "Selesai bila" spec):
  A. deployment_valid (HistoricalScope)          -> guard LOLOS.
  B. legacy tanpa bypass (TargetVisibleScope)     -> guard RAISE (TargetLeakageError).
  C. legacy + bypass eksplisit (allow_target_review=True, hanya sah utk legacy)
                                                  -> guard DILEWATI, run lanjut.

Leave-one-out train juga dibuktikan: eval_index mencampur baris TRAIN dan TEST.
Di bawah HistoricalScope, review milik baris train (u,i) tercabut dari profil
item-nya sendiri -- feature builder toy tidak tahu train vs test, scope yang
mengatur (Invarian #6).

Artefak per protokol ditulis ke checkpoints/results/ dengan prefix
`demo_review_scope_*` (Invarian #4 & #8: prediksi per-baris dipersist).

Jalankan:
    python scripts/demo_review_scope.py --domain amazon_electronics --n-test 200 --n-train 100
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Jalankan dari root repo: pastikan `src` importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config_utils import load_config  # noqa: E402
from src.data.scope import ReviewScope  # noqa: E402
from src.eval.guards import (  # noqa: E402
    TargetLeakageError,
    enforce_no_target_leakage,
)
from src.features.provenance import ProvenanceTracker  # noqa: E402

logger = logging.getLogger("demo_review_scope")

ITEM_COL = "business_id"
USER_COL = "user_id"
RID_COL = "review_id"
RESULTS_DIR = _REPO_ROOT / "checkpoints" / "results"


def _load_eval_and_corpus(domain: str, n_test: int, n_train: int, seed: int):
    """Ambil subset eval (campur test+train) dan bangun korpus review dari split
    asli, dipersempit ke item yang muncul di subset eval agar cepat namun tiap
    item tetap membawa SELURUH review-nya (termasuk review target)."""
    split_dir = _REPO_ROOT / "data" / "splits" / domain
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split {split_dir} tidak ada. Skrip ini memakai split yang SUDAH ada "
            "(Invarian #3), tidak meng-generate ulang."
        )
    train = pd.read_csv(split_dir / "train.csv")
    test = pd.read_csv(split_dir / "test.csv")

    # Subset baris eval: campur TEST dan TRAIN (train utk membuktikan LOO train).
    test_rows = test.sample(n=min(n_test, len(test)), random_state=seed)
    train_rows = train.sample(n=min(n_train, len(train)), random_state=seed)
    eval_rows = pd.concat(
        [test_rows.assign(origin="test"), train_rows.assign(origin="train")],
        ignore_index=True,
    )

    # Korpus = train + test, dipersempit ke item yang dievaluasi. Menyertakan
    # baris test dalam korpus penting: itulah yang membuat review target hadir
    # sebagai kandidat, sehingga TargetVisibleScope benar-benar membocorkannya.
    items = set(eval_rows[ITEM_COL].unique())
    corpus = pd.concat([train, test], ignore_index=True)
    corpus = corpus[corpus[ITEM_COL].isin(items)].reset_index(drop=True)
    return eval_rows, corpus


def _build_toy_features(scope: ReviewScope, eval_rows: pd.DataFrame, corpus: pd.DataFrame):
    """Feature builder TOY: untuk tiap baris (u,i), fitur = rata-rata stars dari
    review item i yang TERLIHAT menurut `scope`. Mencatat provenance & prediksi
    per-baris. row_id = review_id baris (== id review target-nya)."""
    # Index item -> (review_id, stars) sekali, agar loop per-baris O(#review item).
    item_reviews: dict[str, list[tuple[str, float]]] = {}
    for iid, grp in corpus.groupby(ITEM_COL):
        item_reviews[iid] = list(zip(grp[RID_COL], grp["stars"].astype(float)))
    stars_by_rid = dict(zip(corpus[RID_COL], corpus["stars"].astype(float)))
    global_mean = float(corpus["stars"].astype(float).mean())

    prov = ProvenanceTracker()
    eval_index: dict[str, str] = {}   # row_id -> target_review_id
    per_row = []                      # persist per-baris (Invarian #8)

    for r in eval_rows.itertuples(index=False):
        uid = getattr(r, USER_COL)
        iid = getattr(r, ITEM_COL)
        row_id = getattr(r, RID_COL)          # konvensi: row_id == review_id baris
        target_review_id = row_id             # review target baris (u,i)
        ts = getattr(r, "date", None)

        candidates = item_reviews.get(iid, [])
        cand_ids = [rid for rid, _ in candidates]
        visible = scope.filter_visible(cand_ids, uid, iid, ts)

        # Fitur toy = mean stars review yang terlihat (fallback ke global mean
        # kalau kosong -- terjadi di HistoricalScope saat item hanya punya
        # review target itu sendiri).
        vis_stars = [stars_by_rid[rid] for rid in visible]
        feature = sum(vis_stars) / len(vis_stars) if vis_stars else global_mean

        prov.record(row_id, visible)
        eval_index[row_id] = target_review_id
        per_row.append(
            {
                "row_id": str(row_id),
                "user_id": str(uid),
                "business_id": str(iid),
                "origin": getattr(r, "origin", "?"),
                "target_review_id": str(target_review_id),
                "n_candidate_reviews": len(cand_ids),
                "n_visible_reviews": len(visible),
                "target_in_provenance": target_review_id in prov.get(row_id),
                "toy_mean_stars_pred": round(feature, 6),
                "true_stars": float(getattr(r, "stars")),
            }
        )

    return prov, eval_index, per_row


def _persist(domain: str, protocol_name: str, prov: ProvenanceTracker, per_row: list, extra: dict):
    """Tulis prediksi per-baris (Invarian #8) + provenance + ringkasan ke
    checkpoints/results/ dengan prefix demo_review_scope_* (Invarian #4)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"demo_review_scope_{protocol_name}_{domain}"

    pred_csv = RESULTS_DIR / f"{stem}_perrow.csv"
    pd.DataFrame(per_row).to_csv(pred_csv, index=False)

    prov_json = RESULTS_DIR / f"{stem}_provenance.json"
    prov.save_json(prov_json)

    summary = {
        "domain": domain,
        "protocol": protocol_name,
        "n_rows": len(per_row),
        "provenance_summary": prov.summary(),
        "n_rows_with_target_in_provenance": sum(1 for x in per_row if x["target_in_provenance"]),
        **extra,
        "artifacts": {"per_row_csv": pred_csv.name, "provenance_json": prov_json.name},
    }
    summ_yaml = RESULTS_DIR / f"{stem}.yaml"
    with open(summ_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    return summ_yaml, summary


def _run_protocol(config: dict, label: str, domain: str, eval_rows, corpus, expect: str):
    proto = config["protocol"]
    print(f"\n{'='*70}\nSKENARIO {label}  (protocol.name={proto['name']}, "
          f"scope={proto['scope']}, allow_target_review={proto.get('allow_target_review')})")
    print(f"Ekspektasi: {expect}\n{'-'*70}")

    scope = ReviewScope.from_config(config, corpus, item_col=ITEM_COL, user_col=USER_COL)
    prov, eval_index, per_row = _build_toy_features(scope, eval_rows, corpus)

    n_leak_rows = sum(1 for x in per_row if x["target_in_provenance"])
    print(f"Scope terbangun: {scope.name} | baris eval: {len(per_row)} "
          f"(test+train campuran) | baris yg provenance-nya memuat review target: {n_leak_rows}")

    guard_outcome = None
    try:
        enforced = enforce_no_target_leakage(config, prov, eval_index)
        guard_outcome = "enforced_pass" if enforced else "bypassed"
        print(f"GUARD: {'LOLOS (ditegakkan)' if enforced else 'DILEWATI (bypass legacy eksplisit)'}")
    except TargetLeakageError as e:
        guard_outcome = "raised"
        first_line = str(e).splitlines()[0]
        print(f"GUARD: RAISE TargetLeakageError -> {first_line}")

    summ_yaml, _ = _persist(
        domain, proto["name"] + ("_bypass" if label.endswith("C") else ""),
        prov, per_row, extra={"guard_outcome": guard_outcome, "scenario": label},
    )
    print(f"Artefak dipersist: {summ_yaml.relative_to(_REPO_ROOT)} (+ _perrow.csv, _provenance.json)")
    return guard_outcome


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="amazon_electronics")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_rows, corpus = _load_eval_and_corpus(args.domain, args.n_test, args.n_train, args.seed)
    print(f"Domain: {args.domain} | eval rows: {len(eval_rows)} "
          f"({(eval_rows['origin']=='test').sum()} test + {(eval_rows['origin']=='train').sum()} train) "
          f"| korpus (dipersempit ke item eval): {len(corpus)} review")

    cfg_dir = _REPO_ROOT / "configs" / "protocol"
    deployment = load_config(cfg_dir / "deployment_valid.yaml")
    legacy = load_config(cfg_dir / "legacy_target_review.yaml")

    # Skenario B: legacy TANPA bypass -- salin config legacy lalu matikan bypass
    # (protokol tetap lewat config, bukan edit kode: kita cuma menegaskan guard).
    legacy_no_bypass = copy.deepcopy(legacy)
    legacy_no_bypass["protocol"]["allow_target_review"] = False

    outcomes = {}
    outcomes["A"] = _run_protocol(
        deployment, "A", args.domain, eval_rows, corpus,
        expect="HistoricalScope -> review target dikecualikan -> guard LOLOS")
    outcomes["B"] = _run_protocol(
        legacy_no_bypass, "B", args.domain, eval_rows, corpus,
        expect="TargetVisibleScope, bypass MATI -> guard RAISE")
    outcomes["C"] = _run_protocol(
        legacy, "C", args.domain, eval_rows, corpus,
        expect="TargetVisibleScope + bypass legacy eksplisit -> guard DILEWATI")

    print(f"\n{'='*70}\nRINGKASAN")
    ok = (outcomes["A"] == "enforced_pass"
          and outcomes["B"] == "raised"
          and outcomes["C"] == "bypassed")
    for k in ("A", "B", "C"):
        print(f"  Skenario {k}: {outcomes[k]}")
    print(f"KRITERIA 'Selesai bila' TERPENUHI: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
