"""
src/config_utils.py

Util pemuatan config YAML dengan dukungan inheritance opsional via key
`_base: <path relatif thd file config ini>`. Dipakai utk config domain baru
(Amazon/TripAdvisor) supaya tidak perlu menulis ulang seluruh ~90 baris
config per varian (colab/quicktest/absa mode) -- cukup `_base:` + delta
beberapa baris.

100% BACKWARD COMPATIBLE: config TANPA key `_base` (semua config Yelp yang
sudah ada) menghasilkan dict IDENTIK dengan `yaml.safe_load()` polos --
tidak ada regresi ke pipeline yang sudah tervalidasi.

Aturan merge: dict di-merge rekursif key-demi-key (child menang atas base
utk key yang sama), list/scalar diganti penuh (bukan digabung) oleh child.
`_base` mendukung rantai (base config boleh punya `_base` sendiri).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "_base":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """Baca config YAML di `path`, resolve `_base` (jika ada) secara rekursif.

    Path pada `_base` relatif terhadap folder file config yang memuatnya
    (bukan terhadap cwd), supaya config bisa dijalankan dari direktori mana pun.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or not raw.get("_base"):
        return raw

    base_path = (path.parent / raw["_base"]).resolve()
    base_config = load_config(base_path)
    return _deep_merge(base_config, raw)
