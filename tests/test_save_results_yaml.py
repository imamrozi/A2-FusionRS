"""
tests/test_save_results_yaml.py

Regresi utk insiden `protocol_p3_darraz_reimpl_tripadvisor_hotel_seed42.yaml`
(Fase 1 Step 4 smoke test) yang sempat 0 byte -- lihat
`src/evaluation/metrics.py::_to_native` dan `save_results_yaml` utk root
cause & fix (cast numpy -> native + tulis-lalu-rename atomik, Invarian #9).

Root cause PASTI insiden asli tidak terlacak (tidak ada log/traceback
tersimpan dari attempt pertama yang gagal -- gap terpisah yang memotivasi
perbaikan logging di `run_protocol_p2_p3.py`). Test ini karena itu tidak
mereproduksi kegagalan spesifik itu, tapi memverifikasi SELURUH KELAS bug
(tipe numpy tak ter-cast lolos ke yaml.safe_dump) tertutup, plus properti
atomic-write (path asli tidak pernah rusak kalau serialisasi gagal).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import _to_native, save_results_yaml  # noqa: E402


def test_to_native_casts_numpy_scalars_recursively():
    raw = {
        "a_bool": np.bool_(True),
        "an_int": np.int64(42),
        "a_float": np.float64(3.14),
        "nested": {"b": np.float32(1.5), "arr": np.array([1, 2, 3], dtype=np.int64)},
        "a_list": [np.bool_(False), np.int64(7)],
        "already_native": {"x": 1, "y": "text", "z": None},
    }
    native = _to_native(raw)

    assert type(native["a_bool"]) is bool
    assert type(native["an_int"]) is int
    assert type(native["a_float"]) is float
    assert type(native["nested"]["b"]) is float
    assert native["nested"]["arr"] == [1, 2, 3]
    assert all(type(v) is not np.int64 for v in native["nested"]["arr"])
    assert type(native["a_list"][0]) is bool
    assert type(native["a_list"][1]) is int
    assert native["already_native"] == {"x": 1, "y": "text", "z": None}

    # Kriteria inti: hasil HARUS bisa di-yaml.safe_dump tanpa exception --
    # ini yang gagal di insiden asli.
    yaml.safe_dump(native, allow_unicode=True)


def test_save_results_yaml_with_numpy_types_does_not_produce_empty_file(tmp_path):
    """Simulasi persis pola results_summary run_protocol_p2_p3.py: nilai
    campuran numpy (dari .mean()/np.array pandas) & native. Sebelum fix,
    tipe numpy yg lolos tanpa cast eksplisit di suatu caller bisa membuat
    yaml.safe_dump gagal SETELAH file sudah ter-truncate ke 0 byte."""
    summary = {
        "model_name": "protocol_p3_darraz_reimpl",
        "rmse": np.float64(0.9977),
        "mae": np.float64(0.7688),
        "aspect_diagnostics": {
            "pct_aspect_fallback": np.float64(1.933),
            "mean_n_shared_aspects": np.float32(5.239),
        },
        "n_rows_guarded": np.int64(79562),
        "guard_passed": np.bool_(True),
        "provenance_summary": {"total_review_refs": np.int64(123456)},
    }
    path = tmp_path / "protocol_p3_test.yaml"

    save_results_yaml(path, summary)

    assert path.exists()
    assert path.stat().st_size > 0, "YAML tidak boleh 0 byte -- ini persis gejala insiden asli"

    loaded = yaml.safe_load(path.read_text())
    assert loaded["rmse"] == pytest.approx(0.9977)
    assert loaded["n_rows_guarded"] == 79562
    assert loaded["guard_passed"] is True


def test_save_results_yaml_atomic_write_leaves_no_tmp_file(tmp_path):
    """Setelah save_results_yaml sukses, tidak ada file .tmp sisa -- rename
    atomik (os.replace) sudah membersihkannya."""
    path = tmp_path / "atomic_test.yaml"
    save_results_yaml(path, {"rmse": np.float64(1.0)})

    assert path.exists()
    assert not (tmp_path / "atomic_test.yaml.tmp").exists()


def test_save_results_yaml_overwrite_false_raises_instead_of_clobbering(tmp_path):
    """overwrite=False -> penimpaan GAGAL KERAS (FileExistsError) dan file
    LAMA tetap utuh. Dipakai run stage=confirm A2-FusionRS Fase 2 yang
    menyentuh test set & hanya boleh sekali jalan."""
    path = tmp_path / "confirm_result.yaml"
    save_results_yaml(path, {"rmse": np.float64(0.5), "tag": "asli"})

    with pytest.raises(FileExistsError, match="overwrite=False"):
        save_results_yaml(path, {"rmse": np.float64(0.9), "tag": "baru"}, overwrite=False)

    # file lama HARUS tidak tersentuh
    loaded = yaml.safe_load(path.read_text())
    assert loaded["rmse"] == pytest.approx(0.5)
    assert loaded["tag"] == "asli"


def test_save_results_yaml_overwrite_false_ok_when_file_absent(tmp_path):
    """overwrite=False TIDAK menghalangi penulisan pertama kali."""
    path = tmp_path / "fresh.yaml"
    save_results_yaml(path, {"rmse": np.float64(0.7)}, overwrite=False)
    assert yaml.safe_load(path.read_text())["rmse"] == pytest.approx(0.7)


def test_save_results_yaml_overwrite_warns_with_old_and_new_rmse(tmp_path, caplog):
    """Perilaku existing (WAJIB tetap ada -- lihat docstring save_results_yaml
    soal insiden penimpaan senyap sebelumnya): overwrite men-log RMSE lama vs
    baru, TIDAK diam-diam menimpa tanpa jejak."""
    path = tmp_path / "overwrite_test.yaml"
    save_results_yaml(path, {"rmse": np.float64(0.5)})

    with caplog.at_level("WARNING"):
        save_results_yaml(path, {"rmse": np.float64(0.6)})

    assert any("MENIMPA" in rec.message for rec in caplog.records)
