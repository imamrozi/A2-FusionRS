"""
src/eval/guards.py

Fase 1 Step 3 (docs/phase1_spec.md, baris 122-125): gerbang keras anti
target-review leakage, dipanggil SEBAGAI BAGIAN DARI RUN (bukan hanya test
terpisah).

`assert_no_target_leakage(provenance, eval_index)` GAGAL KERAS (raise) bila
review target sebuah baris muncul di provenance baris itu sendiri. Ini pemeriksa
runtime terakhir: kalau sebuah feature builder membocorkan review target (lupa
lewat ReviewScope, atau protokol memang legacy), run berhenti dengan pesan yang
menunjuk baris pelanggarnya.

`enforce_no_target_leakage(config, provenance, eval_index)` adalah pembungkus
sadar-protokol yang dipanggil di akhir pipeline. Ia menegakkan guard untuk
protokol deployment-valid, dan HANYA melewatinya lewat bypass eksplisit yang
terbatas pada arm legacy (`protocol.name == "legacy"` DAN
`protocol.allow_target_review == True`). Kombinasi lain (mis. deployment_valid
mencoba menyalakan bypass) ditolak keras -- bypass tidak bisa bocor jadi default
tersembunyi (Invarian #1 & #5).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class TargetLeakageError(AssertionError):
    """Diangkat ketika review target sebuah baris ikut membangun fitur baris itu."""


def assert_no_target_leakage(provenance, eval_index, *, max_report: int = 20) -> int:
    """Gagal keras bila ada baris yang provenance-nya memuat review targetnya.

    Parameters
    ----------
    provenance : ProvenanceTracker (atau apa pun dengan `.get(row_id) -> set`)
        Catatan review_id yang dipakai per baris.
    eval_index : Mapping[row_id -> target_review_id]
        Untuk tiap baris yang diperiksa, review_id target (review pasangan (u,i)
        yang sedang diprediksi) yang TIDAK BOLEH muncul di provenance baris itu.
        Konvensi proyek: row_id == target_review_id, tapi guard tidak
        mengasumsikannya -- pemetaan eksplisit ini yang menjadi sumber kebenaran.
        Boleh juga berupa iterable review_id (row_id dianggap == review_id).
    max_report : int
        Batas jumlah pelanggaran yang dirinci di pesan exception.

    Returns
    -------
    int : jumlah baris yang diperiksa (kalau lolos).

    Raises
    ------
    TargetLeakageError : bila >=1 baris membocorkan review targetnya.
    """
    if isinstance(eval_index, Mapping):
        items = list(eval_index.items())
    else:
        # iterable review_id -> row_id == review_id
        items = [(rid, rid) for rid in eval_index]

    violations: list[tuple[object, str]] = []
    for row_id, target_review_id in items:
        used = provenance.get(row_id)
        if target_review_id in used:
            violations.append((row_id, target_review_id))

    if violations:
        shown = violations[:max_report]
        detail = "\n".join(
            f"    row_id={row_id!r} memakai review targetnya sendiri "
            f"(review_id={trid!r})"
            for row_id, trid in shown
        )
        more = "" if len(violations) <= max_report else (
            f"\n    ... dan {len(violations) - max_report} pelanggaran lain."
        )
        raise TargetLeakageError(
            f"TARGET-REVIEW LEAKAGE: {len(violations)} dari {len(items)} baris "
            "menurunkan fitur dari review targetnya sendiri.\n"
            f"{detail}{more}\n"
            "Fitur untuk memprediksi (u,i) tidak boleh berasal dari review (u,i) "
            "(Invarian #1). Pastikan feature builder menyaring kandidat lewat "
            "HistoricalScope. Ini sah HANYA di bawah protokol legacy dengan "
            "bypass eksplisit (lihat enforce_no_target_leakage)."
        )
    return len(items)


def enforce_no_target_leakage(config: dict, provenance, eval_index) -> bool:
    """Penegak guard sadar-protokol untuk dipanggil di akhir pipeline.

    Returns
    -------
    bool : True kalau guard ditegakkan dan lolos; False kalau dilewati lewat
           bypass legacy yang eksplisit.

    Raises
    ------
    ValueError         : konfigurasi bypass tidak sah (bypass di luar legacy).
    TargetLeakageError : guard ditegakkan dan menemukan pelanggaran.
    """
    proto = config.get("protocol")
    if not isinstance(proto, dict) or "name" not in proto:
        raise ValueError(
            "Config tidak punya blok `protocol.name`. Sertakan salah satu config "
            "di configs/protocol/ -- protokol tidak pernah dipilih lewat edit kode."
        )
    name = proto["name"]
    allow = bool(proto.get("allow_target_review", False))

    # Bypass hanya sah untuk arm legacy. Blokir keras kombinasi lain supaya
    # tidak ada jalan diam-diam mematikan guard di protokol deployment-valid.
    if allow and name != "legacy":
        raise ValueError(
            f"protocol.allow_target_review=True hanya sah untuk protocol.name "
            f"'legacy', tapi name='{name}'. Bypass guard leakage dilarang di luar "
            "arm legacy (Invarian #1)."
        )

    if allow:
        logger.warning(
            "[BYPASS] Guard target-review leakage DILEWATI: protocol=legacy, "
            "allow_target_review=True. Run ini SENGAJA memakai review target "
            "(mereproduksi protokol lama). Ini tidak sah di luar arm legacy."
        )
        return False

    n = assert_no_target_leakage(provenance, eval_index)
    logger.info(
        "[GUARD] Lolos: %d baris diperiksa, tidak ada yang membocorkan review "
        "targetnya (protocol=%s).", n, name
    )
    return True
