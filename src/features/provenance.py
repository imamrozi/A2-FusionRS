"""
src/features/provenance.py

Fase 1 Step 3 (docs/phase1_spec.md, baris 119-120): pelacakan provenance fitur.

Setiap feature builder (Step 4+) mencatat, untuk setiap baris yang dibangunnya,
review_id mana saja yang IKUT menyusun fitur baris itu. Struktur intinya adalah
`dict[row_id -> set[review_id]]`. Catatan ini kemudian diperiksa guard
`src/eval/guards.py::assert_no_target_leakage`: kalau review target sebuah baris
muncul di provenance-nya sendiri, itu bukti target-review leakage dan run gagal
keras.

Kenapa perlu pelacakan eksplisit (bukan sekadar percaya pada ReviewScope).
ReviewScope adalah gerbang niat ("review ini boleh dipakai"); provenance adalah
catatan fakta ("review ini benar-benar dipakai"). Guard membandingkan fakta
dengan aturan. Kalau suatu builder lupa lewat scope, provenance-nya akan memuat
review target dan guard menangkapnya -- inilah jaring pengaman yang tidak
bergantung pada disiplin tiap builder.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


class ProvenanceTracker:
    """Peta row_id -> himpunan review_id yang dipakai membangun fitur baris itu.

    Dipakai per-builder (satu tracker per feature builder) atau digabung. API
    sengaja minimal supaya murah dipanggil di loop pembangunan fitur.

    Konvensi row_id. Proyek ini memakai `review_id` baris (u,i) sebagai row_id
    -- sama dengan id review target baris tersebut. Dengan begitu guard cukup
    memeriksa apakah `row_id` (== review target) ada di dalam `get(row_id)`.
    Konvensi ini tidak dipaksakan oleh kelas ini (row_id boleh sembarang
    hashable); yang memetakan row_id -> review target adalah `eval_index` yang
    diberikan ke guard.
    """

    def __init__(self) -> None:
        self._map: dict[object, set[str]] = {}

    def record(self, row_id, review_ids: Iterable[str]) -> None:
        """Catat bahwa fitur `row_id` memakai `review_ids`.

        Idempoten dan akumulatif: pemanggilan berulang untuk row_id yang sama
        menyatukan (union) review_id -- aman kalau satu baris dibangun dari
        beberapa sumber fitur (mis. profil aspek item + riwayat user)."""
        bucket = self._map.setdefault(row_id, set())
        bucket.update(review_ids)

    def get(self, row_id) -> set[str]:
        """Salinan himpunan review_id yang tercatat untuk `row_id` (kosong kalau
        belum pernah dicatat). Mengembalikan salinan agar peta internal tidak
        bisa diubah tak sengaja oleh pemanggil."""
        return set(self._map.get(row_id, ()))

    def row_ids(self) -> set:
        return set(self._map.keys())

    def __contains__(self, row_id) -> bool:
        return row_id in self._map

    def __len__(self) -> int:
        return len(self._map)

    def merge(self, other: "ProvenanceTracker") -> "ProvenanceTracker":
        """Gabungkan provenance builder lain ke dalam tracker ini (union per
        row_id). Mengembalikan self agar bisa dirantai."""
        for row_id, ids in other._map.items():
            self.record(row_id, ids)
        return self

    # ---- serialisasi / inspeksi -----------------------------------------

    def to_dict(self) -> dict:
        """Bentuk serializable (row_id str -> list review_id terurut) untuk
        JSON/YAML dan diff yang stabil."""
        return {str(row_id): sorted(ids) for row_id, ids in self._map.items()}

    def summary(self) -> dict:
        """Ringkasan untuk logging/debug: jumlah baris dan statistik jumlah
        review per baris."""
        sizes = [len(ids) for ids in self._map.values()]
        return {
            "n_rows": len(self._map),
            "total_review_refs": sum(sizes),
            "min_reviews_per_row": min(sizes) if sizes else 0,
            "max_reviews_per_row": max(sizes) if sizes else 0,
            "mean_reviews_per_row": (sum(sizes) / len(sizes)) if sizes else 0.0,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "ProvenanceTracker":
        tracker = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for row_id, ids in data.items():
            tracker.record(row_id, ids)
        return tracker
