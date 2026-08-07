"""
src/a2fusionrs/stream_cache.py

Cache untuk keluaran stream DeepMF & CBF (skalar + vektor) per
(domain, seed, stage, config DeepMF/CBF).

MOTIVASI: dalam eksperimen A2-FusionRS Fase 2, beberapa VARIAN AGF
(representation x residual_base x extra_pyabsa x skenario ablasi) dijalankan
pada (domain, seed, stage) yang SAMA. Stream DeepMF/CBF-nya identik untuk
semua varian itu -- keduanya di-fit SEBELUM tahap fusion dan sama sekali
tidak bergantung pada konfigurasi AGF. Padahal justru keduanya yang mahal:
DeepMF OOF melatih 5 model dari nol, CBF LOO merekonstruksi profil item
per baris. Tanpa cache, 4 varian x 3 domain x 3 seed = 36 run mengulang
9 komputasi stream yang sama sebanyak 4x.

KEAMANAN (cache yang salah lebih berbahaya daripada tidak ada cache --
angkanya tetap keluar, tapi diam-diam berasal dari konfigurasi lain):

1. OPT-IN. Default MATI; harus dinyalakan eksplisit lewat --stream-cache.
2. KUNCI LENGKAP. Hash mencakup SEMUA yang mempengaruhi stream: domain,
   seed, stage, dev_fraction, seluruh hyperparameter DeepMF & CBF, serta
   jumlah baris tiap split. Beda satu saja -> kunci beda -> cache miss
   (recompute), bukan hasil salah.
3. VALIDASI SAAT MUAT. Jumlah baris tiap array dicek ulang terhadap
   DataFrame yang sedang diproses; tidak cocok -> cache DIABAIKAN + warning,
   bukan dipaksakan.
4. SIDECAR JSON. Tiap cache punya metadata terbaca-manusia untuk audit
   ("angka ini berasal dari konfigurasi apa").
5. `CACHE_FORMAT_VERSION` -- dinaikkan bila struktur/semantik stream
   berubah, otomatis membatalkan seluruh cache lama.

CATATAN: kalau `CBFConfig.include_sentiment=True`, stream CBF ikut
bergantung pada sumber sentimen (keyword/PyABSA) yang TIDAK termasuk kunci
-> caching DITOLAK untuk kasus itu (lihat `build_cache_key`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Naikkan bila struktur/semantik stream berubah (mis. definisi laten DeepMF
# atau fitur CBF diubah) -- otomatis membatalkan semua cache lama.
CACHE_FORMAT_VERSION = 1

# Nama stream yang di-cache. Urutan tidak penting (disimpan by-key), tapi
# daftarnya WAJIB lengkap: kalau ada stream baru, tambahkan di sini.
STREAM_NAMES = ("deepmf_scalar", "deepmf_latent", "cbf_scalar", "cbf_features")
SPLITS = ("train", "val", "test")

_DEEPMF_KEY_FIELDS = (
    "embedding_dim", "hidden_layers", "dropout", "batch_size", "learning_rate",
    "epochs", "negative_sampling_ratio", "optimizer", "weight_decay",
)
_CBF_KEY_FIELDS = ("method", "pca_components", "include_sentiment", "k_selection")


class StreamCacheUnsafeError(RuntimeError):
    """Cache diminta pada konfigurasi yang kuncinya TIDAK bisa menjamin
    kebenaran (lihat catatan include_sentiment di docstring modul)."""


def build_cache_key(
    config: dict,
    domain: str,
    seed: int,
    stage: str,
    dev_fraction: float | None,
    n_rows: dict[str, int],
) -> tuple[str, dict]:
    """Kembalikan (hash_kunci, payload_metadata).

    Raise `StreamCacheUnsafeError` bila konfigurasi membuat stream
    bergantung pada sesuatu yang TIDAK tercakup kunci.
    """
    cbf_cfg = config.get("cbf_clustering", {})
    if cbf_cfg.get("include_sentiment", False):
        raise StreamCacheUnsafeError(
            "cbf_clustering.include_sentiment=True membuat stream CBF bergantung pada sumber "
            "sentimen (keyword/PyABSA), yang TIDAK termasuk kunci cache -- cache bisa "
            "mengembalikan stream dari sumber sentimen yang berbeda. Matikan --stream-cache "
            "untuk konfigurasi ini."
        )

    deepmf_cfg = config.get("deepmf", {})
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "domain": domain,
        "seed": int(seed),
        "stage": stage,
        "dev_fraction": float(dev_fraction) if (stage == "select" and dev_fraction is not None) else None,
        "deepmf": {k: deepmf_cfg.get(k) for k in _DEEPMF_KEY_FIELDS},
        "cbf": {k: cbf_cfg.get(k) for k in _CBF_KEY_FIELDS},
        "n_rows": {s: int(n_rows[s]) for s in SPLITS},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    key = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    return key, payload


def _npz_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"streams_{key}.npz"


def _json_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"streams_{key}.json"


def load_streams(
    cache_dir: str | Path, key: str, n_rows: dict[str, int]
) -> dict[str, dict[str, np.ndarray]] | None:
    """Muat stream dari cache. Kembalikan None (cache miss) bila file tidak
    ada, tidak lengkap, atau jumlah barisnya tidak cocok -- SELALU aman:
    pemanggil tinggal menghitung ulang."""
    npz_path = _npz_path(cache_dir, key)
    if not npz_path.exists():
        return None

    try:
        with np.load(npz_path) as data:
            streams: dict[str, dict[str, np.ndarray]] = {}
            for name in STREAM_NAMES:
                streams[name] = {}
                for split in SPLITS:
                    flat_key = f"{name}__{split}"
                    if flat_key not in data:
                        logger.warning(
                            "Cache stream %s TIDAK lengkap (hilang '%s') -- dihitung ulang.",
                            npz_path, flat_key,
                        )
                        return None
                    streams[name][split] = data[flat_key]
    except Exception as exc:  # file korup/terpotong
        logger.warning("Cache stream %s gagal dibaca (%s) -- dihitung ulang.", npz_path, exc)
        return None

    # Validasi jumlah baris -- pengaman terakhir kalau ada kunci yang bertabrakan.
    for name in STREAM_NAMES:
        for split in SPLITS:
            actual = len(streams[name][split])
            expected = n_rows[split]
            if actual != expected:
                logger.warning(
                    "Cache stream %s TIDAK cocok: %s[%s] punya %d baris, diharapkan %d "
                    "-- cache DIABAIKAN, stream dihitung ulang.",
                    npz_path, name, split, actual, expected,
                )
                return None

    logger.info("Cache stream DITEMUKAN & valid: %s (DeepMF/CBF tidak dihitung ulang).", npz_path)
    return streams


def save_streams(
    cache_dir: str | Path,
    key: str,
    payload: dict,
    streams: dict[str, dict[str, np.ndarray]],
) -> None:
    """Simpan stream + sidecar metadata. Penulisan .npz dilakukan ke file
    sementara lalu di-rename (atomik) supaya cache tidak pernah setengah
    jadi bila proses terputus di tengah."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    flat = {}
    for name in STREAM_NAMES:
        for split in SPLITS:
            flat[f"{name}__{split}"] = np.asarray(streams[name][split])

    npz_path = _npz_path(cache_dir, key)
    tmp_path = npz_path.with_suffix(".npz.tmp")
    # CATATAN: np.savez() menambahkan '.npz' SECARA OTOMATIS bila diberi
    # PATH yang tidak berakhiran '.npz' -- itu membuat file tmp tersimpan
    # sbg '...npz.tmp.npz' dan rename atomik di bawah gagal (FileNotFound).
    # Ditemukan oleh test_roundtrip_is_bit_identical. Menulis lewat file
    # handle mematikan perilaku auto-append tsb.
    with open(tmp_path, "wb") as f:
        np.savez(f, **flat)
    tmp_path.replace(npz_path)

    with open(_json_path(cache_dir, key), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    total_mb = npz_path.stat().st_size / 1e6
    logger.info("Cache stream disimpan ke %s (%.1f MB) + sidecar metadata.", npz_path, total_mb)
