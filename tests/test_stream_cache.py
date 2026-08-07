"""
tests/test_stream_cache.py

Regresi utk cache stream DeepMF/CBF (src/a2fusionrs/stream_cache.py).

Cache yang SALAH lebih berbahaya daripada tidak ada cache: angkanya tetap
keluar, tapi diam-diam berasal dari konfigurasi lain. Test ini karena itu
fokus pada PENGAMAN, bukan cuma "bisa simpan & muat":
1. Roundtrip: array yang dimuat IDENTIK BIT-PER-BIT dgn yang disimpan.
2. Kunci berubah bila apa pun yang mempengaruhi stream berubah (seed,
   stage, dev_fraction, hyperparameter DeepMF/CBF, jumlah baris).
3. Kunci TIDAK berubah utk konfigurasi yang sama (kalau berubah, cache
   tidak pernah kena -> percuma).
4. Cache miss aman: file hilang/tidak lengkap/korup/jumlah baris beda ->
   None (recompute), BUKAN exception atau data salah.
5. include_sentiment=True DITOLAK (stream bergantung sumber sentimen yang
   tidak tercakup kunci).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.stream_cache import (  # noqa: E402
    SPLITS,
    STREAM_NAMES,
    StreamCacheUnsafeError,
    build_cache_key,
    load_streams,
    save_streams,
)

BASE_CONFIG = {
    "deepmf": {
        "embedding_dim": 128, "hidden_layers": [256, 128, 64, 32], "dropout": 0.3,
        "batch_size": 512, "learning_rate": 0.002, "epochs": 20,
        "negative_sampling_ratio": 0, "optimizer": "adamw", "weight_decay": 0.0,
    },
    "cbf_clustering": {
        "method": "agglomerative", "pca_components": 50,
        "include_sentiment": False, "k_selection": "elbow",
    },
}
N_ROWS = {"train": 100, "val": 20, "test": 30}


def _make_streams(rng_seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.RandomState(rng_seed)
    dims = {"deepmf_scalar": None, "deepmf_latent": 32, "cbf_scalar": None, "cbf_features": 53}
    streams = {}
    for name in STREAM_NAMES:
        streams[name] = {}
        for split in SPLITS:
            n = N_ROWS[split]
            d = dims[name]
            arr = rng.rand(n).astype(np.float32) if d is None else rng.rand(n, d).astype(np.float32)
            streams[name][split] = arr
    return streams


def _key(**overrides):
    cfg = overrides.pop("config", BASE_CONFIG)
    params = {
        "domain": "tripadvisor_hotel", "seed": 42, "stage": "select",
        "dev_fraction": 0.15, "n_rows": N_ROWS,
    }
    params.update(overrides)
    return build_cache_key(cfg, **params)[0]


def test_roundtrip_is_bit_identical(tmp_path):
    """Gerbang WAJIB: array hasil muat harus identik bit-per-bit."""
    streams = _make_streams()
    key, payload = build_cache_key(
        BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS
    )
    save_streams(tmp_path, key, payload, streams)
    loaded = load_streams(tmp_path, key, N_ROWS)

    assert loaded is not None
    for name in STREAM_NAMES:
        for split in SPLITS:
            np.testing.assert_array_equal(loaded[name][split], streams[name][split])
            assert loaded[name][split].dtype == streams[name][split].dtype


def test_sidecar_metadata_written(tmp_path):
    streams = _make_streams()
    key, payload = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)
    save_streams(tmp_path, key, payload, streams)
    assert (tmp_path / f"streams_{key}.json").exists()


def test_key_stable_for_same_config():
    assert _key() == _key(), "Kunci harus deterministik, kalau tidak cache tak pernah kena"


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": 43},
        {"stage": "confirm"},
        {"dev_fraction": 0.20},
        {"domain": "restaurant"},
        {"n_rows": {"train": 101, "val": 20, "test": 30}},
    ],
)
def test_key_changes_when_stream_affecting_param_changes(overrides):
    assert _key(**overrides) != _key(), f"Kunci HARUS berubah untuk {overrides}"


@pytest.mark.parametrize(
    "section,field,new_value",
    [
        ("deepmf", "learning_rate", 0.001),
        ("deepmf", "epochs", 30),
        ("deepmf", "embedding_dim", 64),
        ("deepmf", "optimizer", "sgd"),
        ("deepmf", "negative_sampling_ratio", 4),
        ("cbf_clustering", "method", "kmeans"),
        ("cbf_clustering", "pca_components", 20),
    ],
)
def test_key_changes_when_hyperparameter_changes(section, field, new_value):
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg[section][field] = new_value
    assert _key(config=cfg) != _key(), f"Kunci HARUS berubah untuk {section}.{field}"


def test_cache_miss_when_file_absent(tmp_path):
    assert load_streams(tmp_path, "kunci_tidak_ada", N_ROWS) is None


def test_cache_ignored_when_row_count_mismatch(tmp_path):
    """Pengaman terakhir: jumlah baris beda -> cache DIABAIKAN, bukan dipakai."""
    streams = _make_streams()
    key, payload = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)
    save_streams(tmp_path, key, payload, streams)

    wrong_rows = {"train": 999, "val": 20, "test": 30}
    assert load_streams(tmp_path, key, wrong_rows) is None


def test_cache_miss_when_file_corrupt(tmp_path):
    """File korup -> None (recompute), BUKAN exception yang menghentikan run."""
    key, payload = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)
    save_streams(tmp_path, key, payload, _make_streams())
    (tmp_path / f"streams_{key}.npz").write_bytes(b"bukan npz sama sekali")

    assert load_streams(tmp_path, key, N_ROWS) is None


def test_cache_miss_when_stream_incomplete(tmp_path):
    """npz valid tapi kehilangan satu stream -> None (recompute)."""
    key, payload = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)
    streams = _make_streams()
    partial = {"deepmf_scalar__train": streams["deepmf_scalar"]["train"]}
    np.savez(tmp_path / f"streams_{key}.npz", **partial)

    assert load_streams(tmp_path, key, N_ROWS) is None


def test_include_sentiment_true_is_rejected():
    """Stream CBF jadi bergantung sumber sentimen yang TIDAK ada di kunci."""
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["cbf_clustering"]["include_sentiment"] = True
    with pytest.raises(StreamCacheUnsafeError, match="include_sentiment"):
        build_cache_key(cfg, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)


def test_different_seeds_do_not_share_cache(tmp_path):
    """Uji end-to-end anti-tabrakan: simpan seed 42, muat dgn kunci seed 43
    -> harus miss (bukan mengembalikan stream seed 42)."""
    key42, payload42 = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 42, "select", 0.15, N_ROWS)
    save_streams(tmp_path, key42, payload42, _make_streams(rng_seed=1))

    key43, _ = build_cache_key(BASE_CONFIG, "tripadvisor_hotel", 43, "select", 0.15, N_ROWS)
    assert load_streams(tmp_path, key43, N_ROWS) is None
