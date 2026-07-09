"""
src/baseline/fusion_nmf_dt.py

Reimplementasi tahap fusion final baseline Darraz et al.: kombinasi prediksi
SA + DeepMF + CBF menggunakan NMF (Non-negative Matrix Factorization) untuk
reduksi dimensi, dilanjutkan DecisionTreeRegressor sebagai model prediksi
rating akhir.

PENTING (lih. diskusi anomali RMSE 0.01-0.02 di baseline paper):
Fungsi `evaluate()` di modul ini WAJIB dijalankan pada held-out test set
yang benar-benar terpisah (hasil split_generator.py), BUKAN pada train set.
Jika hasil RMSE reimplementasi ini jauh berbeda (lebih tinggi/realistis)
dibanding angka yang dilaporkan paper asli, itu mengonfirmasi dugaan bahwa
evaluasi asli memiliki masalah metodologis -- dokumentasikan perbandingan
ini secara eksplisit di bagian Discussion draft artikel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    nmf_components: int = 20
    dt_max_depth: int = 10
    random_state: int = 42


class NMFDecisionTreeFusion:
    """Fusion baseline: NMF untuk reduksi dimensi fitur gabungan, lalu
    DecisionTreeRegressor untuk prediksi rating final."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.nmf = NMF(
            n_components=self.config.nmf_components,
            random_state=self.config.random_state,
            init="nndsvda",
            max_iter=500,
        )
        self.dt = DecisionTreeRegressor(
            max_depth=self.config.dt_max_depth, random_state=self.config.random_state
        )
        self._feature_min: np.ndarray | None = None

    def _build_feature_matrix(
        self,
        sentiment_scores: np.ndarray,
        deepmf_preds: np.ndarray,
        cbf_preds: np.ndarray,
    ) -> np.ndarray:
        features = np.stack([sentiment_scores, deepmf_preds, cbf_preds], axis=1)
        return features

    def fit(
        self,
        sentiment_scores: np.ndarray,
        deepmf_preds: np.ndarray,
        cbf_preds: np.ndarray,
        y_true_ratings: np.ndarray,
    ) -> None:
        features = self._build_feature_matrix(sentiment_scores, deepmf_preds, cbf_preds)

        # NMF butuh input non-negatif -- shift jika ada nilai negatif
        # (mis. jika salah satu skor prediksi ternyata bisa negatif)
        self._feature_min = features.min(axis=0)
        features_nonneg = features - np.minimum(self._feature_min, 0)

        nmf_features = self.nmf.fit_transform(features_nonneg)
        self.dt.fit(nmf_features, y_true_ratings)
        logger.info(
            "Fusion NMF+DT dilatih pada %d sampel, %d komponen NMF",
            len(y_true_ratings),
            self.config.nmf_components,
        )

    def predict(
        self,
        sentiment_scores: np.ndarray,
        deepmf_preds: np.ndarray,
        cbf_preds: np.ndarray,
    ) -> np.ndarray:
        if self._feature_min is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict().")
        features = self._build_feature_matrix(sentiment_scores, deepmf_preds, cbf_preds)
        features_nonneg = features - np.minimum(self._feature_min, 0)
        nmf_features = self.nmf.transform(features_nonneg)
        return self.dt.predict(nmf_features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton fusion NMF+DT -- WAJIB dievaluasi pada test set held-out "
        "yang identik dengan model lain, lihat evaluation/metrics.py dan "
        "run_baseline.py untuk orkestrasi penuh."
    )
