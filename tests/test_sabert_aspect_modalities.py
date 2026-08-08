"""
tests/test_sabert_aspect_modalities.py

Regresi untuk jalur fitur BARU di run_attention_gated_fusion.py yang lahir
dari diagnosis Gerbang 1-3 (reports/gates_1_3_summary.md):

  _compute_global_sentiment_modality()   token sentimen global level-review
  _compute_sabert_aspect_rich_modality() order-stats atas skor SA-BERT
  _compute_sabert_aspect_sequences()     sequence aspek + identitas

Yang diverifikasi adalah PROPERTI KEBENARAN, bukan akurasi model:
1. Token global memetakan skor ke baris yang BENAR (bukan sekadar berukuran
   tepat) -- salah-align di sini akan merusak seluruh klaim Gerbang-3 secara
   diam-diam.
2. Baris tanpa skor -> 0,5 DAN memicu warning (bukan lolos senyap).
3. Order-statistics dihitung benar, dicek terhadap perhitungan manual.
4. Review tanpa aspek -> jalur fallback, bukan nol (0 pada skala [0,1]
   berarti "sangat negatif" -- cacat encoding yang pernah mencemari
   Gerbang-2, lihat reports/gates_1_3_summary.md Bagian 2).
5. Vocab aspek dibangun HANYA dari train -> istilah yang cuma ada di test
   dipetakan ke UNK (cegah kebocoran identitas aspek).
6. Truncation menghormati max_aspects; mask konsisten dgn isi.
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

from run_attention_gated_fusion import (  # noqa: E402
    _compute_global_sentiment_modality,
    _compute_sabert_aspect_rich_modality,
    _compute_sabert_aspect_sequences,
)


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    (tmp_path / "sentiment_bert").mkdir(parents=True)
    (tmp_path / "pyabsa").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config(ckpt: Path) -> dict:
    return {"logging": {"checkpoint_dir": str(ckpt)}}


EXP = {"domain": "tripadvisor_hotel"}


# ---------------------------------------------------------------- token global


def test_global_token_maps_scores_to_correct_rows(config, ckpt):
    """Uji ALIGNMENT, bukan sekadar bentuk: urutan review_id di split sengaja
    dibuat BERBEDA dari urutan di file cache."""
    pd.DataFrame({"review_id": [3, 1, 2], "sentiment_score": [0.3, 0.1, 0.2]}).to_csv(
        ckpt / "sentiment_bert" / "sentiment_scores.csv", index=False
    )
    splits = {"train": pd.DataFrame({"review_id": [1, 2, 3]})}

    out = _compute_global_sentiment_modality(config, splits)

    assert out["train"].shape == (3, 1)
    np.testing.assert_allclose(out["train"].ravel(), [0.1, 0.2, 0.3], rtol=1e-6)


def test_global_token_missing_row_warns_and_fills_neutral(config, ckpt, caplog):
    pd.DataFrame({"review_id": [1], "sentiment_score": [0.8]}).to_csv(
        ckpt / "sentiment_bert" / "sentiment_scores.csv", index=False
    )
    splits = {"train": pd.DataFrame({"review_id": [1, 99]})}

    with caplog.at_level("WARNING"):
        out = _compute_global_sentiment_modality(config, splits)

    np.testing.assert_allclose(out["train"].ravel(), [0.8, 0.5], rtol=1e-6)
    # Cakupan cache nyata 100%; kalau cabang ini aktif, HARUS terlihat.
    assert any("TIDAK punya skor" in r.message for r in caplog.records)


def test_global_token_missing_cache_raises(config):
    with pytest.raises(FileNotFoundError, match="run_baseline.py"):
        _compute_global_sentiment_modality(config, {"train": pd.DataFrame({"review_id": [1]})})


# ------------------------------------------------------- rich order-statistics


def _write_aspect_cache(ckpt: Path, rows: list[tuple], fallback: list[tuple]) -> None:
    pd.DataFrame(rows, columns=["review_id", "aspect_term", "sabert_score"]).to_csv(
        ckpt / "pyabsa" / "sabert_aspect_scores_tripadvisor_hotel.csv", index=False
    )
    pd.DataFrame(fallback, columns=["review_id", "fallback_score"]).to_csv(
        ckpt / "pyabsa" / "sabert_fallback_tripadvisor_hotel.csv", index=False
    )


def test_rich_order_statistics_match_manual(config, ckpt):
    _write_aspect_cache(
        ckpt,
        [(1, "room", 0.9), (1, "staff", 0.1), (1, "food", 0.6)],
        [],
    )
    splits = {"train": pd.DataFrame({"review_id": [1]})}

    out = _compute_sabert_aspect_rich_modality(config, EXP, splits)[["train"][0]]
    s = np.array([0.9, 0.1, 0.6])

    assert out.shape == (1, 9)
    np.testing.assert_allclose(out[0, 0], min(3 / 3.0, 1.0), rtol=1e-5)   # n_aspects_norm
    np.testing.assert_allclose(out[0, 1], s.mean(), rtol=1e-5)            # mean_pos
    np.testing.assert_allclose(out[0, 2], s.min(), rtol=1e-5)             # min_pos
    np.testing.assert_allclose(out[0, 3], s.max(), rtol=1e-5)             # max_pos
    np.testing.assert_allclose(out[0, 4], s.max() - s.min(), rtol=1e-5)   # range
    np.testing.assert_allclose(out[0, 5], (1 - s).max(), rtol=1e-5)       # max_neg
    np.testing.assert_allclose(out[0, 7], (s < 0.5).mean(), rtol=1e-5)    # frac_negative
    np.testing.assert_allclose(out[0, 8], (s >= 0.5).mean(), rtol=1e-5)   # frac_positive


def test_rich_no_aspect_uses_fallback_not_zero(config, ckpt):
    """Regresi terhadap cacat encoding Gerbang-2: pada skala [0,1], mengisi
    0 berarti 'sangat negatif', BUKAN 'tidak ada data'."""
    _write_aspect_cache(ckpt, [(1, "room", 0.9)], [(2, 0.75)])
    splits = {"train": pd.DataFrame({"review_id": [1, 2]})}

    out = _compute_sabert_aspect_rich_modality(config, EXP, splits)["train"]

    assert out[1, 0] == 0.0                                   # tidak ada bukti aspek
    np.testing.assert_allclose(out[1, 1], 0.75, rtol=1e-5)    # mean = fallback
    np.testing.assert_allclose(out[1, 5], 0.25, rtol=1e-5)    # max_neg = 1 - fallback
    assert out[1, 1] != 0.0, "baris fallback tidak boleh dikodekan sebagai 0"


def test_rich_missing_cache_points_to_precompute(config):
    with pytest.raises(FileNotFoundError, match="precompute_pyabsa_sabert_scores"):
        _compute_sabert_aspect_rich_modality(
            config, EXP, {"train": pd.DataFrame({"review_id": [1]})}
        )


# ------------------------------------------------------------------- sequences


def test_vocab_built_from_train_only_test_terms_become_unk(config, ckpt):
    _write_aspect_cache(
        ckpt,
        [(1, "room", 0.9), (2, "rahasia_test", 0.2)],
        [],
    )
    splits = {
        "train": pd.DataFrame({"review_id": [1]}),
        "test": pd.DataFrame({"review_id": [2]}),
    }

    vocab, out = _compute_sabert_aspect_sequences(config, EXP, splits)

    assert "room" in vocab
    assert "rahasia_test" not in vocab, "istilah khusus test bocor ke vocab"
    assert out["test"]["ids"][0, 0] == 1, "istilah tak dikenal harus -> UNK (1)"


def test_sequence_features_and_mask_consistent(config, ckpt):
    _write_aspect_cache(ckpt, [(1, "room", 0.8), (1, "staff", 0.25)], [])
    splits = {"train": pd.DataFrame({"review_id": [1]})}

    _, out = _compute_sabert_aspect_sequences(config, EXP, splits)
    feats, mask = out["train"]["feats"], out["train"]["mask"]

    assert mask[0, :2].all() and not mask[0, 2:].any()
    np.testing.assert_allclose(feats[0, 0], [1 - 0.8, 0.0, 0.8, abs(0.8 - 0.5) * 2], rtol=1e-5)
    np.testing.assert_allclose(feats[0, 1], [1 - 0.25, 0.0, 0.25, abs(0.25 - 0.5) * 2], rtol=1e-5)
    assert not feats[0, 2:].any(), "slot padding harus nol"


def test_sequence_truncates_at_max_aspects(config, ckpt):
    _write_aspect_cache(ckpt, [(1, f"a{i}", 0.5 + i / 100) for i in range(12)], [])
    splits = {"train": pd.DataFrame({"review_id": [1]})}

    _, out = _compute_sabert_aspect_sequences(config, EXP, splits, max_aspects=8)

    assert out["train"]["ids"].shape[1] == 8
    assert out["train"]["mask"][0].sum() == 8


def test_sequence_review_without_aspect_gets_fallback_slot(config, ckpt):
    _write_aspect_cache(ckpt, [(1, "room", 0.9)], [(2, 0.4)])
    splits = {"train": pd.DataFrame({"review_id": [1, 2]})}

    _, out = _compute_sabert_aspect_sequences(config, EXP, splits)

    assert out["train"]["mask"][1, 0], "review fallback harus tetap punya 1 slot aktif"
    assert out["train"]["ids"][1, 0] == 1, "slot fallback memakai UNK"
    np.testing.assert_allclose(out["train"]["feats"][1, 0, 2], 0.4, rtol=1e-5)
