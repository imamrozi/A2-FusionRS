"""
src/baseline/fusion_nmf_dt.py

Reimplementasi tahap fusion final baseline Darraz et al.: kombinasi prediksi
SA + DeepMF + CBF menggunakan NMF (Non-negative Matrix Factorization) untuk
reduksi dimensi, dilanjutkan DecisionTreeRegressor sebagai model prediksi
rating akhir.

Mengikuti Algorithm 1 (Section 3.4.3) paper persis:
1. Fitur mentah = (DeepMF_predictions, cluster/CBF_predictions, BERT_Predictions)
   -- 3 kolom di baseline asli, sehingga n_components NMF idealnya <= 3
   (paper pakai contoh 3).
2. NMF diterapkan ke fitur mentah untuk menghasilkan fitur laten.
3. Fitur laten NMF di-CONCATENATE dengan fitur mentah asli (bukan
   menggantikannya) -- lihat `_build_feature_matrix` & `fit()`/`predict()`
   di bawah, langkah "Combine Original Features with NMF Features".
4. DecisionTreeRegressor dilatih pada fitur gabungan tsb.

GENERALISASI (utk varian ablasi ABSA-concat): `sentiment_scores` boleh 1D
(n,) -- 1 skor per baris, seperti SA global/ABSA-mean -- ATAU 2D (n,k) --
k skor terpisah per baris, mis. skor per-aspek ABSA TANPA agregasi jadi 1
angka. `run_baseline.py` (SA global) tetap kirim 1D, tidak ada perubahan
perilaku sama sekali di sana.

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
    # Fitur mentah hanya 3 kolom (DeepMF, CBF/cluster, BERT sentiment) --
    # NMF tidak bisa menghasilkan komponen lebih banyak dari jumlah fitur
    # input. Paper (Algorithm 1) memakai contoh n_components=3.
    nmf_components: int = 3
    dt_max_depth: int = 10
    random_state: int = 42


class NMFDecisionTreeFusion:
    """Fusion baseline: NMF untuk reduksi dimensi fitur gabungan, lalu
    DecisionTreeRegressor untuk prediksi rating final."""

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        # NMF DIBANGUN DI fit() (bukan di sini) -- baru tahu jumlah fitur
        # mentah sebenarnya (3 utk SA global/ABSA-mean, atau lebih utk
        # ABSA-concat) setelah lihat data, supaya n_components bisa di-cap
        # otomatis (lihat fit()) -- cegah error "n_components > n_features"
        # seperti yang pernah terjadi sebelumnya.
        self.nmf: NMF | None = None
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
        # sentiment_scores boleh 1D (n,) -- 1 skor/baris -- atau 2D (n,k) --
        # k skor terpisah/baris (mis. per-aspek ABSA tanpa agregasi).
        sentiment_2d = (
            sentiment_scores.reshape(-1, 1) if sentiment_scores.ndim == 1 else sentiment_scores
        )
        deepmf_2d = deepmf_preds.reshape(-1, 1)
        cbf_2d = cbf_preds.reshape(-1, 1)
        features = np.concatenate([sentiment_2d, deepmf_2d, cbf_2d], axis=1)
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

        n_components = min(self.config.nmf_components, features.shape[0] - 1, features.shape[1])
        n_components = max(n_components, 1)
        if n_components != self.config.nmf_components:
            logger.warning(
                "nmf_components di config (%d) melebihi batas aman untuk %d fitur mentah -- "
                "di-cap otomatis ke %d.",
                self.config.nmf_components,
                features.shape[1],
                n_components,
            )
        self.nmf = NMF(
            n_components=n_components,
            random_state=self.config.random_state,
            init="nndsvda",
            max_iter=500,
        )

        nmf_features = self.nmf.fit_transform(features_nonneg)
        # Algorithm 1 langkah 3 (paper): gabungkan fitur asli dengan fitur
        # laten NMF, JANGAN buang fitur aslinya.
        combined_features = np.concatenate([features, nmf_features], axis=1)
        self.dt.fit(combined_features, y_true_ratings)
        logger.info(
            "Fusion NMF+DT dilatih pada %d sampel, %d fitur mentah, %d komponen NMF (+%d fitur "
            "asli = %d kolom input DT)",
            len(y_true_ratings),
            features.shape[1],
            n_components,
            features.shape[1],
            combined_features.shape[1],
        )

    def predict(
        self,
        sentiment_scores: np.ndarray,
        deepmf_preds: np.ndarray,
        cbf_preds: np.ndarray,
    ) -> np.ndarray:
        if self._feature_min is None or self.nmf is None:
            raise RuntimeError("Panggil fit() terlebih dahulu sebelum predict().")
        features = self._build_feature_matrix(sentiment_scores, deepmf_preds, cbf_preds)
        features_nonneg = features - np.minimum(self._feature_min, 0)
        nmf_features = self.nmf.transform(features_nonneg)
        combined_features = np.concatenate([features, nmf_features], axis=1)
        return self.dt.predict(combined_features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton fusion NMF+DT -- WAJIB dievaluasi pada test set held-out "
        "yang identik dengan model lain, lihat evaluation/metrics.py dan "
        "run_baseline.py untuk orkestrasi penuh."
    )
