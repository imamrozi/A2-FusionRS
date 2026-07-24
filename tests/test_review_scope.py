"""
tests/test_review_scope.py

Fase 1 Step 3 -- regresi untuk infrastruktur ReviewScope + ProvenanceTracker +
guard. Sebagian besar assertion memakai korpus sintetis kecil (deterministik,
cepat). Satu test memakai SLICE split asli (Invarian #3: tidak me-regenerate
split) untuk membuktikan jalur bekerja pada data riil; test itu SKIP kalau split
tidak tersedia, tidak pernah pura-pura lolos.

Kontrak yang diuji:
  - HistoricalScope mengecualikan review target (u,i); TargetVisibleScope tidak.
  - Leave-one-out berlaku identik untuk baris train maupun eval (Invarian #6).
  - Guard RAISE saat review target masuk provenance; LOLOS saat tidak.
  - enforce_no_target_leakage: deployment_valid menegakkan; legacy+bypass
    melewati; bypass di luar legacy ditolak keras.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.scope import HistoricalScope, TargetVisibleScope, build_scope, ReviewScope
from src.eval.guards import (
    TargetLeakageError,
    assert_no_target_leakage,
    enforce_no_target_leakage,
)
from src.features.provenance import ProvenanceTracker

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_corpus() -> pd.DataFrame:
    # Item i1 punya 3 review (dari u1,u2,u3); item i2 punya 2 (u1,u2).
    # review_id memakai konvensi proyek u::i::epoch.
    rows = [
        ("u1::i1::1", "u1", "i1", 5.0, 1),
        ("u2::i1::2", "u2", "i1", 3.0, 2),
        ("u3::i1::3", "u3", "i1", 4.0, 3),
        ("u1::i2::4", "u1", "i2", 2.0, 4),
        ("u2::i2::5", "u2", "i2", 1.0, 5),
    ]
    return pd.DataFrame(rows, columns=["review_id", "user_id", "business_id", "stars", "date"])


def test_target_visible_includes_target():
    scope = TargetVisibleScope(_synthetic_corpus())
    vis = scope.visible_review_ids("u1", "i1")
    assert "u1::i1::1" in vis  # review target TERLIHAT


def test_historical_excludes_target_only():
    scope = HistoricalScope(_synthetic_corpus())
    vis = scope.visible_review_ids("u1", "i1")
    assert "u1::i1::1" not in vis          # target dicabut
    assert "u2::i1::2" in vis              # review item lain tetap ada (reputasi item)
    assert "u1::i2::4" in vis              # review user di item lain tetap ada (riwayat user)


def test_filter_visible_matches_is_visible():
    scope = HistoricalScope(_synthetic_corpus())
    item_i1 = ["u1::i1::1", "u2::i1::2", "u3::i1::3"]
    filtered = scope.filter_visible(item_i1, "u1", "i1")
    assert filtered == {"u2::i1::2", "u3::i1::3"}


def test_leave_one_out_symmetric_train_and_eval():
    """LOO harus bergantung pada pasangan (u,i), bukan pada label train/eval.
    Query untuk (u2,i1) mencabut review (u2,i1) baik baris itu 'train' maupun
    'test' -- HistoricalScope tidak tahu bedanya (Invarian #6)."""
    scope = HistoricalScope(_synthetic_corpus())
    for uid in ("u1", "u2", "u3"):
        vis = scope.visible_review_ids(uid, "i1")
        assert scope.target_review_ids(uid, "i1").isdisjoint(vis)


def test_guard_raises_when_target_in_provenance():
    prov = ProvenanceTracker()
    prov.record("u1::i1::1", {"u1::i1::1", "u2::i1::2"})  # memuat review targetnya
    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(prov, {"u1::i1::1": "u1::i1::1"})


def test_guard_passes_when_target_excluded():
    prov = ProvenanceTracker()
    prov.record("u1::i1::1", {"u2::i1::2", "u3::i1::3"})  # target dikecualikan
    n = assert_no_target_leakage(prov, {"u1::i1::1": "u1::i1::1"})
    assert n == 1


def test_enforce_deployment_valid_enforced():
    prov = ProvenanceTracker()
    prov.record("u1::i1::1", {"u2::i1::2"})
    cfg = {"protocol": {"name": "deployment_valid", "scope": "historical", "allow_target_review": False}}
    assert enforce_no_target_leakage(cfg, prov, {"u1::i1::1": "u1::i1::1"}) is True


def test_enforce_legacy_bypass_skips_guard():
    prov = ProvenanceTracker()
    prov.record("u1::i1::1", {"u1::i1::1"})  # bocor, tapi bypass legacy aktif
    cfg = {"protocol": {"name": "legacy", "scope": "target_visible", "allow_target_review": True}}
    assert enforce_no_target_leakage(cfg, prov, {"u1::i1::1": "u1::i1::1"}) is False


def test_enforce_bypass_outside_legacy_rejected():
    prov = ProvenanceTracker()
    cfg = {"protocol": {"name": "deployment_valid", "scope": "historical", "allow_target_review": True}}
    with pytest.raises(ValueError):
        enforce_no_target_leakage(cfg, prov, {"u1::i1::1": "u1::i1::1"})


def test_provenance_serialization_roundtrip(tmp_path):
    prov = ProvenanceTracker()
    prov.record("r1", {"a", "b"})
    prov.record("r1", {"b", "c"})  # akumulatif (union)
    assert prov.get("r1") == {"a", "b", "c"}
    p = prov.save_json(tmp_path / "prov.json")
    reloaded = ProvenanceTracker.load_json(p)
    assert reloaded.get("r1") == {"a", "b", "c"}


def test_build_scope_from_config_registry():
    corpus = _synthetic_corpus()
    cfg = {"protocol": {"name": "deployment_valid", "scope": "historical"}}
    scope = ReviewScope.from_config(cfg, corpus)
    assert isinstance(scope, HistoricalScope)
    assert isinstance(build_scope("target_visible", corpus), TargetVisibleScope)


@pytest.mark.parametrize("domain", ["amazon_electronics", "yelp_restaurant", "tripadvisor_hotel"])
def test_real_split_slice_end_to_end(domain):
    """Bukti pada data riil: HistoricalScope -> guard lolos; TargetVisibleScope
    -> guard raise. Pakai slice kecil split asli (Invarian #3)."""
    split_dir = _REPO_ROOT / "data" / "splits" / domain
    test_csv = split_dir / "test.csv"
    train_csv = split_dir / "train.csv"
    if not test_csv.exists() or not train_csv.exists():
        pytest.skip(f"Split {domain} tidak tersedia di mesin ini.")

    test = pd.read_csv(test_csv)
    train = pd.read_csv(train_csv)
    if test.empty:
        pytest.skip(f"Split test {domain} kosong.")

    eval_rows = test.sample(n=min(50, len(test)), random_state=42)
    items = set(eval_rows["business_id"].unique())
    corpus = pd.concat([train, test], ignore_index=True)
    corpus = corpus[corpus["business_id"].isin(items)].reset_index(drop=True)

    item_ids = {iid: list(g["review_id"]) for iid, g in corpus.groupby("business_id")}

    def build(scope):
        prov = ProvenanceTracker()
        idx = {}
        for r in eval_rows.itertuples(index=False):
            rid = r.review_id
            vis = scope.filter_visible(item_ids.get(r.business_id, []), r.user_id, r.business_id)
            prov.record(rid, vis)
            idx[rid] = rid
        return prov, idx

    hist_prov, idx = build(HistoricalScope(corpus))
    assert assert_no_target_leakage(hist_prov, idx) == len(idx)  # lolos

    tv_prov, idx2 = build(TargetVisibleScope(corpus))
    with pytest.raises(TargetLeakageError):
        assert_no_target_leakage(tv_prov, idx2)  # raise


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
