"""
src/baseline/classical_cf.py

Baseline collaborative-filtering RINGAN (Item-KNN, SVD klasik) sebagai
pembanding kompleksitas/akurasi terhadap DeepMF -- MURNI CF, tanpa sentiment
analysis maupun content-based, TIDAK terintegrasi ke pipeline fusion
(run_baseline.py). Dijalankan & dievaluasi berdiri sendiri lewat
run_classical_cf.py, memakai split identik dengan baseline hybrid.

Memakai scikit-surprise, library CF standar. Nama "CoClustering" di Table
8/9 paper Darraz et al. persis nama kelas di library ini -- indikasi kuat
paper aslinya juga memakai surprise, jadi dipakai di sini juga untuk fidelity
perbandingan (meski CoClustering sendiri belum diimplementasikan -- lihat
`algorithm` sebagai string-dispatch yang mudah diperluas nanti).

CATATAN DESAIN PENTING: berbeda dari deepmf.py/cbf_clustering.py, modul ini
TIDAK memakai mapping user2idx/item2idx manual -- surprise.Trainset menyimpan
mapping raw-id<->inner-id sendiri, dan algo.predict(raw_uid, raw_iid) terima
raw id langsung. Ini deviasi yang DISENGAJA dari konvensi modul lain di repo
ini, bukan inkonsistensi.

Cold-start (user/item tak dikenal saat training) ditangani otomatis oleh
surprise: algo.predict() fallback ke trainset.global_mean, ditandai
prediction.details["was_impossible"]=True -- kita cuma log jumlahnya,
konsisten dengan gaya n_fallback logging di cbf_clustering.py/deepmf.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.trainset import Trainset

logger = logging.getLogger(__name__)


def _build_trainset(df: pd.DataFrame, rating_scale: tuple[float, float]) -> Trainset:
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["user_id", "business_id", "stars"]], reader)
    return data.build_full_trainset()


@dataclass
class ClassicalCFConfig:
    algorithm: str = "item_knn"  # "item_knn" | "svd"
    # --- Item-KNN ---
    knn_k: int = 40
    knn_min_k: int = 1
    knn_sim_name: str = "cosine"  # "cosine" | "pearson" | "msd"
    knn_user_based: bool = False  # False = item-based (Item-KNN)
    # --- SVD ---
    svd_n_factors: int = 100
    svd_n_epochs: int = 20
    svd_lr_all: float = 0.005
    svd_reg_all: float = 0.02
    random_state: int = 42


class ClassicalCFTrainer:
    """Wrapper fit/predict untuk Item-KNN atau SVD (surprise), API konsisten
    dengan Trainer/Predictor lain di repo ini (fit(train_df) -> predict(df))."""

    def __init__(self, config: ClassicalCFConfig | None = None):
        self.config = config or ClassicalCFConfig()
        if self.config.algorithm == "item_knn":
            self._algo = KNNBasic(
                k=self.config.knn_k,
                min_k=self.config.knn_min_k,
                sim_options={
                    "name": self.config.knn_sim_name,
                    "user_based": self.config.knn_user_based,
                },
                verbose=False,
            )
        elif self.config.algorithm == "svd":
            self._algo = SVD(
                n_factors=self.config.svd_n_factors,
                n_epochs=self.config.svd_n_epochs,
                lr_all=self.config.svd_lr_all,
                reg_all=self.config.svd_reg_all,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(
                f"algorithm '{self.config.algorithm}' tidak dikenal -- gunakan 'item_knn' atau 'svd'"
            )
        self._trainset: Trainset | None = None

    def fit(self, train_df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)) -> None:
        self._trainset = _build_trainset(train_df, rating_scale)
        self._algo.fit(self._trainset)
        logger.info(
            "%s dilatih pada %d interaksi (%d user, %d item)",
            self.config.algorithm,
            self._trainset.n_ratings,
            self._trainset.n_users,
            self._trainset.n_items,
        )

    def predict(self, df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)) -> np.ndarray:
        if self._trainset is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict().")

        preds = np.empty(len(df), dtype=np.float32)
        n_fallback = 0
        for idx, row in enumerate(df.itertuples(index=False)):
            prediction = self._algo.predict(str(row.user_id), str(row.business_id))
            preds[idx] = prediction.est
            if prediction.details.get("was_impossible"):
                n_fallback += 1

        if n_fallback > 0:
            logger.info(
                "%d/%d baris memakai fallback global_mean (user/item baru) saat prediksi %s",
                n_fallback,
                len(df),
                self.config.algorithm,
            )
        return np.clip(preds, rating_scale[0], rating_scale[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton classical_cf -- jalankan via run_classical_cf.py setelah split "
        "tersedia (data/splits/<domain>/), lihat UserBasedSplitGenerator.load()."
    )
