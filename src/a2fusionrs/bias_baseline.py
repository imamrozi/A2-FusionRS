"""
src/a2fusionrs/bias_baseline.py

Baseline bias klasik matrix factorization (mu + b_u + b_i, gaya Koren 2009)
sebagai JANGKAR (base residual) untuk Attention-Gated Fusion.

MOTIVASI ARSITEKTURAL: tanpa jangkar, AGF harus meregresi SELURUH level
rating dari nol lewat `sigmoid(head_out)` -- ia tidak punya prior "rata-rata
rating" yang didapat DecisionTree secara gratis dari nilai daunnya. Versi
lama A2-FusionRS mengatasinya dengan memakai prediksi NMF+DecisionTree
sebagai base, TAPI itu kontradiktif secara naratif: komponen itulah yang
justru diklaim digantikan AGF. Bias baseline menyelesaikan masalah yang
sama TANPA kontradiksi -- ia prior statistik standar di literatur MF, bukan
mekanisme fusi saingan.

DUA PENGAMAN ANTI-LEAKAGE (konsisten dgn disiplin OOF/LOO proyek ini --
lihat compute_oof_predictions_with_latent() di src/baseline/deepmf.py dan
CBFPredictor.predict_train_loo()):

1. DAMPING/SHRINKAGE (lambda): b_u = sum(r - mu) / (n_u + lambda). Tanpa
   damping, user/item dgn 1-2 rating mendapat bias ekstrem yang murni noise.
   lambda DITETAPKAN A PRIORI (=10, nilai lazim di literatur) dan SENGAJA
   TIDAK di-tune -- menambah axis tuning berarti menambah risiko p-hacking,
   padahal jangkar ini cuma alat bantu, bukan objek studi.

2. LEAVE-ONE-OUT utk baris TRAIN: bias yang dipakai menilai baris (u,i) di
   train dihitung TANPA menyertakan rating baris itu sendiri. Tanpa ini,
   base train jadi optimistik (baris ikut membentuk biasnya sendiri) ->
   residual train mengecil semu -> AGF belajar mengoreksi sinyal yang tidak
   akan ada saat test. LOO di sini ANALITIK O(N) (bukan refit per baris):
   cukup kurangi kontribusi baris ybs dari akumulator.

Baris val/test TIDAK butuh LOO -- rating mereka memang tidak pernah ikut
membentuk bias (bias hanya di-fit dari train), jadi `predict()` biasa sudah
genuinely out-of-sample.

User/item yang tidak dikenal (cold-start) -> bias 0, jatuh ke mu global.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Damping default. Ditetapkan a priori, JANGAN di-tune (lihat docstring).
DEFAULT_BIAS_DAMPING = 10.0


class UserItemBiasBaseline:
    """Prediktor rating mu + b_u + b_i dgn damping & koreksi LOO utk train.

    API sengaja meniru `CBFPredictor` (`fit`/`predict`/`predict_train_loo`)
    supaya pemanggilan di runner konsisten dgn stream lain.
    """

    def __init__(self, damping: float = DEFAULT_BIAS_DAMPING):
        if damping < 0:
            raise ValueError(f"damping harus >= 0 (diberikan {damping}).")
        self.damping = float(damping)
        self.global_mean: float | None = None
        # Akumulator mentah disimpan (bukan cuma bias final) karena koreksi
        # LOO analitik butuh sum & count per user/item.
        self._user_sum: dict = {}
        self._user_count: dict = {}
        self._item_sum: dict = {}
        self._item_count: dict = {}
        self._user_bias: dict = {}
        self._item_bias: dict = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        """Hitung mu, b_u, b_i dari TRAIN saja (urutan standar: b_u dulu,
        lalu b_i atas residual setelah b_u -- Koren 2009)."""
        ratings = train_df["stars"].to_numpy(dtype=np.float64)
        users = train_df["user_id"].to_numpy()
        items = train_df["business_id"].to_numpy()

        self.global_mean = float(ratings.mean())

        # --- b_u dari residual thd mu ---
        dev_u = pd.DataFrame({"user_id": users, "dev": ratings - self.global_mean})
        grp_u = dev_u.groupby("user_id")["dev"].agg(["sum", "count"])
        self._user_sum = grp_u["sum"].to_dict()
        self._user_count = grp_u["count"].to_dict()
        self._user_bias = {
            u: s / (self._user_count[u] + self.damping) for u, s in self._user_sum.items()
        }

        # --- b_i dari residual thd (mu + b_u) ---
        u_bias_arr = np.array([self._user_bias.get(u, 0.0) for u in users], dtype=np.float64)
        dev_i = pd.DataFrame(
            {"business_id": items, "dev": ratings - self.global_mean - u_bias_arr}
        )
        grp_i = dev_i.groupby("business_id")["dev"].agg(["sum", "count"])
        self._item_sum = grp_i["sum"].to_dict()
        self._item_count = grp_i["count"].to_dict()
        self._item_bias = {
            i: s / (self._item_count[i] + self.damping) for i, s in self._item_sum.items()
        }

        logger.info(
            "UserItemBiasBaseline di-fit: mu=%.4f, %d user bias, %d item bias, damping=%.1f",
            self.global_mean, len(self._user_bias), len(self._item_bias), self.damping,
        )

    def _check_fitted(self) -> None:
        if self.global_mean is None:
            raise RuntimeError("Panggil fit() terlebih dahulu.")

    def predict(
        self, df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)
    ) -> np.ndarray:
        """Prediksi utk baris val/test (TIDAK perlu LOO -- rating mereka
        tidak pernah ikut membentuk bias)."""
        self._check_fitted()
        rating_min, rating_max = rating_scale
        u_bias = df["user_id"].map(self._user_bias).fillna(0.0).to_numpy(dtype=np.float64)
        i_bias = df["business_id"].map(self._item_bias).fillna(0.0).to_numpy(dtype=np.float64)
        preds = self.global_mean + u_bias + i_bias
        return np.clip(preds, rating_min, rating_max).astype(np.float32)

    def predict_train_loo(
        self, train_df: pd.DataFrame, rating_scale: tuple[float, float] = (1.0, 5.0)
    ) -> np.ndarray:
        """Prediksi utk baris TRAIN dgn koreksi leave-one-out ANALITIK:
        kontribusi baris ybs dikeluarkan dari mu, b_u, dan b_i.

        Catatan: b_i LOO memakai b_u yang SUDAH ter-LOO (konsisten dgn
        urutan perhitungan di fit(): b_i dihitung atas residual setelah b_u).
        Item/user dgn hanya 1 rating -> penyebut (n-1+lambda) tetap > 0
        berkat damping, jadi TIDAK ada pembagian nol dan biasnya menyusut ke
        0 secara alami (persis perilaku yang diinginkan utk data tipis).
        """
        self._check_fitted()
        rating_min, rating_max = rating_scale

        ratings = train_df["stars"].to_numpy(dtype=np.float64)
        users = train_df["user_id"].to_numpy()
        items = train_df["business_id"].to_numpy()
        n_total = len(ratings)

        # --- mu LOO ---
        total_sum = ratings.sum()
        mu_loo = (
            (total_sum - ratings) / (n_total - 1)
            if n_total > 1
            else np.full(n_total, self.global_mean, dtype=np.float64)
        )

        # --- b_u LOO (kontribusi baris ini dikeluarkan) ---
        u_sum = np.array([self._user_sum.get(u, 0.0) for u in users], dtype=np.float64)
        u_cnt = np.array([self._user_count.get(u, 0) for u in users], dtype=np.float64)
        own_dev_u = ratings - self.global_mean
        u_bias_loo = (u_sum - own_dev_u) / (u_cnt - 1 + self.damping)

        # --- b_i LOO (pakai b_u LOO, konsisten urutan fit()) ---
        i_sum = np.array([self._item_sum.get(i, 0.0) for i in items], dtype=np.float64)
        i_cnt = np.array([self._item_count.get(i, 0) for i in items], dtype=np.float64)
        own_dev_i = ratings - self.global_mean - u_bias_loo
        i_bias_loo = (i_sum - own_dev_i) / (i_cnt - 1 + self.damping)

        preds = mu_loo + u_bias_loo + i_bias_loo
        logger.info(
            "UserItemBiasBaseline.predict_train_loo: %d baris train diprediksi dgn koreksi LOO "
            "(mu/b_u/b_i semua mengecualikan baris ybs).",
            n_total,
        )
        return np.clip(preds, rating_min, rating_max).astype(np.float32)
