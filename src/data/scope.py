"""
src/data/scope.py

Fase 1 Step 3 (docs/phase1_spec.md, baris 107-137): definisi TUNGGAL aturan
visibilitas review di seluruh codebase.

Motivasi. Lineage Darraz et al. -> A2-IRM -> A2-FusionRS menurunkan fitur untuk
memprediksi rating pasangan (user u, item i) dari teks review d_ui itu sendiri.
Saat deployment nyata d_ui belum ditulis (user belum mencoba item), jadi fitur
apa pun yang diturunkan darinya adalah target-review leakage. Aturan
"boleh/tidak boleh memakai review X untuk memprediksi (u,i)" TERLALU mudah
dilanggar tanpa sengaja kalau tersebar ke banyak feature builder. Karena itu
seluruh feature builder (Step 4+) WAJIB menanyakan modul ini review mana yang
boleh dipakai -- tidak ada modul lain yang mendefinisikan aturan ini.

Dua protokol (dipilih lewat config `configs/protocol/*.yaml`, TIDAK PERNAH lewat
edit kode -- Invarian #5):

  * TargetVisibleScope  -> review target d_ui TERLIHAT (protokol lama/legacy).
  * HistoricalScope     -> review target d_ui DIKECUALIKAN (deployment-valid).

Leave-one-out untuk baris TRAIN (Invarian #6 / "jebakan implementasi" spec).
`visible_review_ids`/`filter_visible` bersifat per-baris: dipanggil dengan
(user_id, item_id) dari baris yang sedang dibangun fiturnya -- baik baris TRAIN
maupun EVAL. HistoricalScope mengecualikan review (u,i) berdasarkan pasangan
(user,item) query, BUKAN berdasarkan apakah baris itu train atau test. Jadi bila
sebuah baris train (u,i) ikut membangun "profil aspek historis item i", review
(u,i) itu otomatis tercabut dari agregat -- feature builder tidak perlu tahu
soal train-vs-test sama sekali. Ini yang membuat leave-one-out train benar
secara default, bukan sebagai kasus khusus yang gampang terlupa.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Nama scope yang valid di config `protocol.scope`. Dipetakan ke kelas di
# `build_scope`. Menaruh registry di sini (bukan di config_utils) menjaga
# invariant: aturan visibilitas hidup HANYA di modul ini.
_SCOPE_REGISTRY = {
    "target_visible": "TargetVisibleScope",
    "historical": "HistoricalScope",
}


class ReviewScope(ABC):
    """Gerbang visibilitas review. Satu instance membungkus satu korpus review
    (biasanya train + baris yang sedang dievaluasi) dan menjawab, untuk sebuah
    query (user_id, item_id, timestamp): review mana yang BOLEH dipakai untuk
    membangun fitur guna memprediksi pasangan (user_id, item_id) itu.

    Kontrak yang dijamin subclass:
      - `target_review_ids(u, i)` mengembalikan review_id milik pasangan (u,i)
        di korpus (biasanya satu; bisa >1 kalau user mereview item sama berkali).
      - `is_visible(review_id, u, i, ts)` -> bool: apakah review_id boleh dipakai
        untuk query (u,i). O(1), dipakai per-kandidat oleh feature builder.
      - `visible_review_ids(u, i, ts)` -> set[str]: seluruh review yang boleh
        dipakai (spec Step 3). Materialisasi penuh korpus; untuk pemakaian
        produktif per-baris lebih hemat pakai `filter_visible`.
      - `filter_visible(candidate_ids, u, i, ts)` -> set[str]: subset dari
        `candidate_ids` yang boleh dipakai. O(len(candidate_ids)). INI yang
        dipanggil feature builder (mis. saring review milik item i saja).

    Parameters
    ----------
    reviews : pd.DataFrame
        Korpus review. Wajib punya kolom `review_id`, `user_id`, dan kolom item
        (`item_col`). Kolom timestamp opsional (untuk penegakan temporal).
    item_col : str
        Nama kolom item. Default "business_id" (skema semua domain proyek ini).
    timestamp_col : Optional[str]
        Nama kolom timestamp (mis. "date"). Hanya dipakai kalau penegakan
        temporal diaktifkan oleh subclass.
    """

    def __init__(
        self,
        reviews: pd.DataFrame,
        item_col: str = "business_id",
        user_col: str = "user_id",
        review_id_col: str = "review_id",
        timestamp_col: Optional[str] = "date",
    ) -> None:
        for col in (review_id_col, user_col, item_col):
            if col not in reviews.columns:
                raise ValueError(
                    f"Korpus review wajib punya kolom '{col}'. Kolom tersedia: "
                    f"{list(reviews.columns)}"
                )
        self._item_col = item_col
        self._user_col = user_col
        self._review_id_col = review_id_col
        self._timestamp_col = timestamp_col

        # review_id -> (user_id, item_id, timestamp)
        self._meta: dict[str, tuple[str, str, object]] = {}
        # (user_id, item_id) -> frozenset[review_id]  (identifikasi review target)
        self._target_index: dict[tuple[str, str], set[str]] = {}

        has_ts = timestamp_col is not None and timestamp_col in reviews.columns
        cols = [review_id_col, user_col, item_col] + ([timestamp_col] if has_ts else [])
        for row in reviews[cols].itertuples(index=False, name=None):
            rid, uid, iid = row[0], row[1], row[2]
            ts = row[3] if has_ts else None
            self._meta[rid] = (uid, iid, ts)
            self._target_index.setdefault((uid, iid), set()).add(rid)

        # Cache set seluruh review_id korpus (dipakai visible_review_ids).
        self._all_ids: frozenset[str] = frozenset(self._meta.keys())

    # ---- API publik ------------------------------------------------------

    @property
    def name(self) -> str:
        return type(self).__name__

    def target_review_ids(self, user_id: str, item_id: str) -> set[str]:
        """review_id milik pasangan (user_id, item_id) di korpus (review target)."""
        return set(self._target_index.get((user_id, item_id), ()))

    @abstractmethod
    def is_visible(self, review_id: str, user_id: str, item_id: str, timestamp=None) -> bool:
        """Apakah `review_id` boleh dipakai untuk memprediksi (user_id, item_id)?"""

    def visible_review_ids(self, user_id: str, item_id: str, timestamp=None) -> set[str]:
        """Seluruh review di korpus yang boleh dipakai untuk query (u,i).

        Ini tanda tangan yang diminta spec. Untuk korpus besar dan pemakaian
        per-baris, `filter_visible` jauh lebih hemat (tidak memindai korpus
        penuh). Default: saring seluruh korpus lewat `is_visible`; subclass
        boleh override untuk jalur cepat.
        """
        return {
            rid
            for rid in self._all_ids
            if self.is_visible(rid, user_id, item_id, timestamp)
        }

    def filter_visible(
        self,
        candidate_ids: Iterable[str],
        user_id: str,
        item_id: str,
        timestamp=None,
    ) -> set[str]:
        """Subset `candidate_ids` yang boleh dipakai untuk query (u,i).

        Jalur yang DISARANKAN untuk feature builder: berikan hanya kandidat yang
        relevan (mis. review milik item i, atau review milik user u), biar biaya
        O(len(candidate_ids)) bukan O(korpus)."""
        return {
            rid
            for rid in candidate_ids
            if self.is_visible(rid, user_id, item_id, timestamp)
        }

    # ---- factory ---------------------------------------------------------

    @staticmethod
    def from_config(config: dict, reviews: pd.DataFrame, **kwargs) -> "ReviewScope":
        """Bangun scope dari dict config yang punya blok `protocol`.

        `protocol.scope` menentukan implementasi ("target_visible"/"historical").
        Protokol dipilih HANYA lewat config (Invarian #5) -- kode pemanggil tidak
        pernah menyebut nama kelas scope secara literal.
        """
        proto = config.get("protocol")
        if not isinstance(proto, dict) or "scope" not in proto:
            raise ValueError(
                "Config tidak punya blok `protocol.scope`. Sertakan salah satu "
                "config di configs/protocol/ (mis. via `_base:`), jangan pilih "
                "protokol lewat edit kode."
            )
        return build_scope(proto["scope"], reviews, **kwargs)


class TargetVisibleScope(ReviewScope):
    """Protokol lama/legacy: review target d_ui TERLIHAT.

    Setiap review di korpus boleh dipakai, TERMASUK review (u,i) itu sendiri.
    Ini mereproduksi kondisi yang menghasilkan angka ledger (protokol
    target-review). Run di bawah scope ini SEHARUSNYA memicu guard
    `assert_no_target_leakage` -- itulah gunanya: menandai bahwa protokol ini
    memang membocorkan review target. Guard hanya boleh dilewati lewat bypass
    eksplisit yang terbatas pada arm legacy (lihat src/eval/guards.py).
    """

    def is_visible(self, review_id: str, user_id: str, item_id: str, timestamp=None) -> bool:
        return review_id in self._meta


class HistoricalScope(ReviewScope):
    """Protokol deployment-valid: review target d_ui DIKECUALIKAN.

    Sebuah review boleh dipakai untuk query (u,i) kecuali ia adalah review
    pasangan (u,i) itu sendiri. Berlaku identik untuk baris train maupun eval,
    sehingga leave-one-out otomatis benar di dalam split train (Invarian #6):
    ketika baris train (u,i) ikut membangun profil historis item i, review (u,i)
    tercabut dari agregat tanpa feature builder perlu membedakan train vs test.

    Penegakan temporal (opsional, default MATI). Fase 1 utama (P1/P2/P3) memakai
    split user-based, bukan temporal -- pengecualian target sudah cukup dan
    menjaga komparabilitas antar-arm. `enforce_temporal=True` (dipakai Step 6,
    robustness temporal) tambahan mengecualikan review yang timestamp-nya >=
    timestamp query.
    """

    def __init__(self, *args, enforce_temporal: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enforce_temporal = enforce_temporal

    def is_visible(self, review_id: str, user_id: str, item_id: str, timestamp=None) -> bool:
        meta = self._meta.get(review_id)
        if meta is None:
            return False
        r_user, r_item, r_ts = meta
        # Cabut review target (u,i) itu sendiri.
        if r_user == user_id and r_item == item_id:
            return False
        # Opsional: cabut review yang belum ada pada saat query (temporal).
        if self._enforce_temporal and timestamp is not None and r_ts is not None:
            try:
                if r_ts >= timestamp:
                    return False
            except TypeError:
                # Tipe timestamp tak terbanding -- jangan diam-diam meloloskan.
                raise TypeError(
                    "enforce_temporal=True tapi timestamp query dan timestamp "
                    f"review tak terbanding ({type(timestamp)!r} vs {type(r_ts)!r}). "
                    "Samakan tipe (mis. keduanya pd.Timestamp atau epoch int)."
                )
        return True

    def visible_review_ids(self, user_id: str, item_id: str, timestamp=None) -> set[str]:
        # Jalur cepat kalau temporal mati: seluruh korpus minus review target.
        if not self._enforce_temporal:
            return set(self._all_ids) - self.target_review_ids(user_id, item_id)
        return super().visible_review_ids(user_id, item_id, timestamp)


def build_scope(scope_name: str, reviews: pd.DataFrame, **kwargs) -> ReviewScope:
    """Instansiasi scope dari nama string (`protocol.scope` di config)."""
    if scope_name not in _SCOPE_REGISTRY:
        raise ValueError(
            f"protocol.scope '{scope_name}' tidak dikenal. Pilihan: "
            f"{sorted(_SCOPE_REGISTRY)}."
        )
    cls = globals()[_SCOPE_REGISTRY[scope_name]]
    return cls(reviews, **kwargs)
