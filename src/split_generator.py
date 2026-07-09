"""
src/split_generator.py

Modul ini menghasilkan split train/val/test yang IDENTIK untuk semua model
yang dibandingkan (baseline reimplementasi, SVD, NCF, DeepFM, A2-FusionRS,
dan seluruh varian ablasi). Split disimpan sebagai file agar tidak perlu
di-generate ulang setiap eksperimen -- ini krusial untuk validitas
komparasi (lih. diskusi fair comparison sebelumnya).

Strategi split: user-based, bukan random-row, untuk mencegah leakage
interaksi user yang sama antara train dan test. Cold-start user/item
disisihkan secara terpisah sebagai skenario evaluasi tambahan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitPaths:
    train: Path
    val: Path
    test: Path
    cold_start: Path


class UserBasedSplitGenerator:
    """Generate split train/val/test per user dengan seed tetap.

    Untuk tiap user, interaksinya diurutkan berdasarkan waktu (`date`) lalu
    dibagi proporsional -- ini mensimulasikan skenario realistis (prediksi
    interaksi masa depan berdasarkan histori), bukan random split murni yang
    bisa membocorkan informasi temporal.

    Parameters
    ----------
    train_ratio, val_ratio, test_ratio : float
        Harus berjumlah 1.0
    seed : int
        Seed acak. WAJIB sama persis di semua eksperimen yang dibandingkan.
    cold_start_holdout : bool
        Jika True, user/item yang jumlah interaksinya terlalu sedikit untuk
        displit proporsional akan dipindah ke set cold_start terpisah,
        bukan dipaksa masuk ke train/test.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        cold_start_holdout: bool = True,
        min_interactions_for_split: int = 3,
    ):
        ratios_sum = round(train_ratio + val_ratio + test_ratio, 6)
        if ratios_sum != 1.0:
            raise ValueError(f"train+val+test ratio harus = 1.0, dapat {ratios_sum}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.cold_start_holdout = cold_start_holdout
        self.min_interactions_for_split = min_interactions_for_split
        self._rng = np.random.default_rng(seed)

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if "date" not in df.columns:
            raise ValueError(
                "Kolom 'date' diperlukan untuk time-aware split per user. "
                "Jika tidak tersedia, gunakan random split dan dokumentasikan "
                "keterbatasan ini secara eksplisit di paper."
            )

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        train_rows, val_rows, test_rows, cold_start_rows = [], [], [], []

        for user_id, group in df.groupby("user_id"):
            group = group.sort_values("date")
            n = len(group)

            if n < self.min_interactions_for_split:
                if self.cold_start_holdout:
                    cold_start_rows.append(group)
                else:
                    # fallback: seluruh interaksi user langsung masuk train
                    train_rows.append(group)
                continue

            n_train = max(1, int(round(n * self.train_ratio)))
            n_val = max(1, int(round(n * self.val_ratio))) if n - n_train > 1 else 0
            n_train = min(n_train, n - 1)  # sisakan minimal 1 untuk test

            train_part = group.iloc[:n_train]
            remaining = group.iloc[n_train:]

            if n_val > 0 and len(remaining) > n_val:
                val_part = remaining.iloc[:n_val]
                test_part = remaining.iloc[n_val:]
            else:
                val_part = remaining.iloc[0:0]  # kosong
                test_part = remaining

            train_rows.append(train_part)
            if not val_part.empty:
                val_rows.append(val_part)
            test_rows.append(test_part)

        # PENTING: fallback untuk split kosong HARUS mempertahankan skema
        # kolom asli (df.iloc[0:0]), BUKAN pd.DataFrame() kosong tanpa kolom.
        # DataFrame tanpa kolom akan tersimpan sebagai file CSV benar-benar
        # kosong (tanpa header), yang menyebabkan pandas.errors.EmptyDataError
        # saat dibaca ulang oleh load(). Ini terutama rawan terjadi pada
        # subset kecil/quicktest di mana banyak user persis di batas
        # min_interactions_for_split, sehingga val_part bisa kosong untuk
        # SEMUA user sekaligus.
        empty_with_schema = df.iloc[0:0]

        result = {
            "train": pd.concat(train_rows).reset_index(drop=True) if train_rows else empty_with_schema,
            "val": pd.concat(val_rows).reset_index(drop=True) if val_rows else empty_with_schema,
            "test": pd.concat(test_rows).reset_index(drop=True) if test_rows else empty_with_schema,
            "cold_start": pd.concat(cold_start_rows).reset_index(drop=True) if cold_start_rows else empty_with_schema,
        }

        for name, part in result.items():
            if part.empty:
                logger.warning(
                    "Split '%s' hasilnya KOSONG (0 baris). Ini valid secara "
                    "teknis (tidak akan error saat load), tapi kemungkinan "
                    "menandakan data terlalu sedikit/homogen untuk strategi "
                    "split saat ini -- pertimbangkan menurunkan "
                    "min_interactions_for_split atau menambah ukuran subset.",
                    name,
                )

        self._validate_no_leakage(result)
        self._log_summary(result)
        return result

    def _validate_no_leakage(self, splits: dict[str, pd.DataFrame]) -> None:
        """Sanity check: pastikan tidak ada review_id yang muncul di lebih dari
        satu split. Ini adalah pengaman terakhir, bukan pengganti desain split
        yang benar."""
        seen_ids: set = set()
        for name in ["train", "val", "test", "cold_start"]:
            ids = set(splits[name].get("review_id", pd.Series(dtype=object)))
            overlap = seen_ids & ids
            if overlap:
                raise RuntimeError(
                    f"LEAKAGE TERDETEKSI: {len(overlap)} review_id muncul di "
                    f"'{name}' dan split sebelumnya. Split generator memiliki bug."
                )
            seen_ids |= ids

    def _log_summary(self, splits: dict[str, pd.DataFrame]) -> None:
        for name, part in splits.items():
            logger.info(
                "Split '%s': %d baris, %d user unik, %d item unik",
                name,
                len(part),
                part["user_id"].nunique() if not part.empty else 0,
                part["business_id"].nunique() if not part.empty else 0,
            )

    def save(self, splits: dict[str, pd.DataFrame], output_dir: str | Path) -> SplitPaths:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, part in splits.items():
            path = output_dir / f"{name}.csv"
            part.to_csv(path, index=False)
            paths[name] = path
            logger.info("Split '%s' disimpan ke %s (%d baris)", name, path, len(part))

        # Simpan metadata seed & konfigurasi split untuk audit trail/reproducibility
        meta_path = output_dir / "split_meta.txt"
        with open(meta_path, "w") as f:
            f.write(f"seed={self.seed}\n")
            f.write(f"train_ratio={self.train_ratio}\n")
            f.write(f"val_ratio={self.val_ratio}\n")
            f.write(f"test_ratio={self.test_ratio}\n")
            f.write(f"cold_start_holdout={self.cold_start_holdout}\n")

        return SplitPaths(
            train=paths["train"],
            val=paths["val"],
            test=paths["test"],
            cold_start=paths["cold_start"],
        )

    @staticmethod
    def load(output_dir: str | Path) -> dict[str, pd.DataFrame]:
        """Load split yang sudah pernah disimpan -- dipakai oleh semua model
        pembanding agar dijamin memakai split identik."""
        output_dir = Path(output_dir)
        result = {}
        for name in ["train", "val", "test", "cold_start"]:
            path = output_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Split '{name}' tidak ditemukan di {path}. "
                    "Jalankan split_generator.py terlebih dahulu."
                )
            try:
                result[name] = pd.read_csv(path)
            except pd.errors.EmptyDataError as e:
                raise RuntimeError(
                    f"File split '{name}' di {path} benar-benar kosong (tanpa "
                    "header sama sekali) -- ini file sisa dari bug versi lama "
                    "split_generator.py yang sudah diperbaiki. HAPUS folder "
                    f"'{output_dir}' secara keseluruhan lalu jalankan ulang "
                    "run_baseline.py agar split di-generate ulang dengan "
                    "kode yang sudah benar."
                ) from e
        return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import YelpDatasetLoader  # noqa: E402

    loader = YelpDatasetLoader("data/raw/yelp_training_set_review.csv", domain="restaurant")
    data = loader.load()
    data = loader.filter_min_interactions(data, min_reviews_per_user=5, min_reviews_per_item=5)

    generator = UserBasedSplitGenerator(seed=42, cold_start_holdout=True)
    splits = generator.split(data)
    generator.save(splits, output_dir="data/splits/yelp_restaurant")
