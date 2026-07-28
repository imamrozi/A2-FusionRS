"""
diagnose_no_sentiment_floor.py

Debug kenapa protokol 'no_sentiment_ablation' (kolom sentimen dibuat
konstan nol, dipaksa lewat NMF(n_components=3) tetap) menghasilkan RMSE
1.5-2.1 -- jauh lebih buruk dari VARIAN ABSA TERLEMAH sekalipun (yang
tetap punya sinyal sentimen, walau buruk). Hipotesis: NMF dg 1 dari 3
kolom konstan (rank-deficient input, n_components == n_features) jadi
tidak stabil, & fitur laten NMF yang rusak itu ikut dipakai
DecisionTreeRegressor (fusion_nmf_dt.py menggabungkan fitur mentah +
fitur laten NMF).

Cara kerja: jalankan run_pipeline() SEKALI (protokol no_sentiment_ablation,
domain hotel -- domain terkecil, cache BERT/CBF sudah ada lokal), tapi
monkeypatch NMFDecisionTreeFusion.fit/predict utk MEREKAM argumen asli
(sentiment_scores, deepmf_preds, cbf_preds, y_true) alih2 cuma menjalankan
fusion apa adanya. Dari situ, offline (tanpa perlu re-run DeepMF/CBF),
bandingkan 3 skenario fusion:
  (a) real         -- sentimen asli (verifikasi angka baseline cocok)
  (b) zero_const_3 -- sentimen konstan nol, TAPI TETAP nmf_components=3
                       (persis kode no_sentiment_ablation saat ini)
  (c) dropped_2     -- kolom sentimen DIHAPUS SELURUHNYA dari feature
                       matrix, nmf_components diturunkan ke 2 (bukan
                       dipaksa 3 pada matriks rank-deficient)

Usage:
    python scripts/diagnose_no_sentiment_floor.py
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from sklearn.ensemble import RandomForestRegressor  # noqa: F401 (import guard, unused)
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import NMF

from src.baseline.fusion_nmf_dt import FusionConfig, NMFDecisionTreeFusion
from src.config_utils import load_config
from src.evaluation.metrics import compute_rmse_mae
import run_baseline_absa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CAPTURED: dict = {}

_orig_fit = NMFDecisionTreeFusion.fit
_orig_predict = NMFDecisionTreeFusion.predict


def _capturing_fit(self, sentiment_scores, deepmf_preds, cbf_preds, y_true_ratings):
    CAPTURED["train_sentiment"] = np.array(sentiment_scores, copy=True)
    CAPTURED["train_deepmf"] = np.array(deepmf_preds, copy=True)
    CAPTURED["train_cbf"] = np.array(cbf_preds, copy=True)
    CAPTURED["y_train"] = np.array(y_true_ratings, copy=True)
    CAPTURED["fusion_config"] = copy.deepcopy(self.config)
    return _orig_fit(self, sentiment_scores, deepmf_preds, cbf_preds, y_true_ratings)


def _capturing_predict(self, sentiment_scores, deepmf_preds, cbf_preds):
    CAPTURED["test_sentiment"] = np.array(sentiment_scores, copy=True)
    CAPTURED["test_deepmf"] = np.array(deepmf_preds, copy=True)
    CAPTURED["test_cbf"] = np.array(cbf_preds, copy=True)
    return _orig_predict(self, sentiment_scores, deepmf_preds, cbf_preds)


def build_features(sentiment, deepmf, cbf, drop_sentiment: bool) -> np.ndarray:
    sentiment_2d = sentiment.reshape(-1, 1) if sentiment.ndim == 1 else sentiment
    deepmf_2d = deepmf.reshape(-1, 1)
    cbf_2d = cbf.reshape(-1, 1)
    if drop_sentiment:
        return np.concatenate([deepmf_2d, cbf_2d], axis=1)
    return np.concatenate([sentiment_2d, deepmf_2d, cbf_2d], axis=1)


def run_fusion_variant(name: str, drop_sentiment: bool, n_components: int, seed: int) -> float:
    train_sent = CAPTURED["train_sentiment"]
    test_sent = CAPTURED["test_sentiment"]

    train_feats = build_features(train_sent, CAPTURED["train_deepmf"], CAPTURED["train_cbf"], drop_sentiment)
    test_feats = build_features(test_sent, CAPTURED["test_deepmf"], CAPTURED["test_cbf"], drop_sentiment)

    feat_min = train_feats.min(axis=0)
    train_nonneg = train_feats - np.minimum(feat_min, 0)
    test_nonneg = test_feats - np.minimum(feat_min, 0)

    nmf = NMF(n_components=n_components, random_state=seed, init="nndsvda", max_iter=500)
    train_nmf = nmf.fit_transform(train_nonneg)
    test_nmf = nmf.transform(test_nonneg)

    train_combined = np.concatenate([train_feats, train_nmf], axis=1)
    test_combined = np.concatenate([test_feats, test_nmf], axis=1)

    dt = DecisionTreeRegressor(max_depth=CAPTURED["fusion_config"].dt_max_depth, random_state=seed)
    dt.fit(train_combined, CAPTURED["y_train"])
    preds = np.clip(dt.predict(test_combined), 1.0, 5.0)

    rmse, mae = compute_rmse_mae(CAPTURED["y_test"], preds)
    logger.info("[%s] n_components=%d drop_sentiment=%s -> RMSE=%.4f MAE=%.4f", name, n_components, drop_sentiment, rmse, mae)
    return rmse


def main() -> None:
    config = load_config("configs/tripadvisor_hotel_config_absa_concat_confidence.yaml")
    config["experiment"]["seed"] = 42

    NMFDecisionTreeFusion.fit = _capturing_fit
    NMFDecisionTreeFusion.predict = _capturing_predict
    try:
        # Jalankan protokol REAL (target_review) dulu -- ini juga merekam
        # train/test streams DeepMF+CBF yang IDENTIK dgn yg dipakai floor
        # (config sama, hanya sentiment_protocol yg beda), sekaligus
        # memverifikasi angka "real" cocok dgn Table III (RMSE hotel
        # concat+confidence ~0.628).
        logger.info("=== Menjalankan pipeline REAL (target_review) utk capture streams ===")
        run_baseline_absa.run_pipeline(config, sentiment_protocol="target_review")
    finally:
        NMFDecisionTreeFusion.fit = _orig_fit
        NMFDecisionTreeFusion.predict = _orig_predict

    # y_test tidak otomatis kecapture (predict() tidak menerima y_true) --
    # ambil dari CBF/DeepMF split yang sama via load_config+split loader
    # sekali lagi, jalur pendek: baca ulang split.
    from pathlib import Path

    from src.split_generator import UserBasedSplitGenerator

    splits = UserBasedSplitGenerator.load(Path(config["split"]["output_dir"]))
    CAPTURED["y_test"] = splits["test"]["stars"].values

    seed = config["experiment"]["seed"]
    configured_n_components = config["fusion_baseline"]["nmf_components"]

    logger.info("=" * 70)
    logger.info("SKENARIO (a) real sentiment, n_components=%d (baseline utk verifikasi)", configured_n_components)
    run_fusion_variant("real", drop_sentiment=False, n_components=configured_n_components, seed=seed)

    # Ganti sentiment jadi konstan nol -- REPLIKASI PERSIS kode
    # no_sentiment_ablation saat ini (n_components tetap 3, dipaksa pada
    # matriks rank-deficient: kolom sentimen constant=0).
    CAPTURED["train_sentiment"] = np.zeros(len(CAPTURED["y_train"]), dtype=np.float32)
    CAPTURED["test_sentiment"] = np.zeros(len(CAPTURED["y_test"]), dtype=np.float32)

    logger.info("=" * 70)
    logger.info("SKENARIO (b) sentiment KONSTAN NOL, n_components=3 (KODE SAAT INI)")
    run_fusion_variant("zero_const_3comp", drop_sentiment=False, n_components=3, seed=seed)

    logger.info("=" * 70)
    logger.info("SKENARIO (c) sentiment DIHAPUS, n_components=2 (kandidat perbaikan)")
    run_fusion_variant("dropped_2comp", drop_sentiment=True, n_components=2, seed=seed)

    logger.info("=" * 70)
    logger.info("SKENARIO (d) sentiment konstan nol TAPI n_components diturunkan ke 2")
    run_fusion_variant("zero_const_2comp", drop_sentiment=False, n_components=2, seed=seed)


if __name__ == "__main__":
    main()
