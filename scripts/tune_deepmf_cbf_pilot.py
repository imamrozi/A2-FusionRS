"""
scripts/tune_deepmf_cbf_pilot.py

Pilot hyperparameter tuning utk DeepMF & CBF (1 domain, beberapa kombinasi
kecil, val RMSE) -- lihat diskusi: di bawah framing "arsitektur terinspirasi
Darraz dengan penyesuaian" (bukan reimplementation literal), hyperparameter
yang cuma diwariskan dari paper (tanpa validasi thd domain Amazon/Hotel yg
struktur datanya beda dari Yelp restoran asli) jadi kelemahan metodologis --
lihat catatan negative_sampling_ratio di docstring deepmf.py sbg bukti
hyperparameter memang sensitif per-domain.

CAKUPAN SENGAJA DIBATASI (pilot, bukan search penuh):
- 1 domain (tripadvisor_hotel -- paling murah).
- DeepMF: 6 kombinasi tangan-pilih (bukan grid penuh) di sekitar setting
  default (embedding_dim/hidden_layers/dropout/learning_rate).
- CBF: 3 kombinasi pca_components (method clustering & fitur lain TETAP,
  sesuai domain -- hotel = agglomerative, bukan bagian yg ditala di sini).
- HANYA train+val dipakai (val utk skoring, model selection) -- test_df
  TIDAK PERNAH dimuat/dipreprocess sama sekali di script ini, supaya tidak
  ada risiko kontak dgn test set walau tidak sengaja.
- sentiment_score diisi KONSTAN dummy (0.5) utk train_df -- CBF butuh kolom
  ini ada scr struktural (build_item_dataframe() selalu meng-agregasi
  sentiment_agg walau CBFConfig.include_sentiment=False mengabaikannya di
  _numeric_cols()) TAPI TIDAK DIPAKAI sbg fitur (default include_sentiment
  =False) -- jadi TIDAK PERLU load model BERT/keyword scorer sama sekali,
  mempercepat pilot ini signifikan.

KETERBATASAN YANG DISADARI (didokumentasikan eksplisit, bukan disembunyikan):
val set yang sama dipakai baik utk early-stopping DI DALAM tiap fit DeepMF
MAUPUN utk memilih config TERBAIK ANTAR kombinasi -- secara teoritis idealnya
pakai dev-set terpisah dari val. Split yang ada cuma 80/10/10 (train/val/
test), tidak ada fold ke-4. Ini simplifikasi yg disengaja utk pilot murah;
kalau hasil pilot ini dipakai utk klaim final di manuskrip, pertimbangkan
nested validation atau k-fold pada train+val gabungan.

Usage:
    python scripts/tune_deepmf_cbf_pilot.py --config configs/tripadvisor_hotel_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.cbf_clustering import CBFConfig, CBFPredictor
from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset
from src.config_utils import load_config
from src.evaluation.metrics import compute_rmse_mae
from src.preprocessing import TextPreprocessor
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


DEEPMF_CANDIDATES: list[dict] = [
    {"name": "default", "embedding_dim": 128, "hidden_layers": (256, 128, 64, 32), "dropout": 0.3, "learning_rate": 0.001},
    {"name": "smaller_embedding", "embedding_dim": 64, "hidden_layers": (128, 64, 32), "dropout": 0.3, "learning_rate": 0.001},
    {"name": "lower_dropout", "embedding_dim": 128, "hidden_layers": (256, 128, 64, 32), "dropout": 0.2, "learning_rate": 0.001},
    {"name": "higher_dropout", "embedding_dim": 128, "hidden_layers": (256, 128, 64, 32), "dropout": 0.5, "learning_rate": 0.001},
    {"name": "higher_lr", "embedding_dim": 128, "hidden_layers": (256, 128, 64, 32), "dropout": 0.3, "learning_rate": 0.005},
    {"name": "wider", "embedding_dim": 128, "hidden_layers": (512, 256, 128, 64), "dropout": 0.3, "learning_rate": 0.001},
]

CBF_PCA_CANDIDATES: list[int] = [20, 30, 50, 70, 90, 110, 130, 160, 200]


def tune_deepmf(train_df, val_df, user2idx, item2idx, n_items, seed, rating_scale) -> list[dict]:
    results = []
    for cand in DEEPMF_CANDIDATES:
        torch.manual_seed(seed)
        config = DeepMFConfig(
            embedding_dim=cand["embedding_dim"],
            hidden_layers=cand["hidden_layers"],
            dropout=cand["dropout"],
            batch_size=512,
            learning_rate=cand["learning_rate"],
            negative_sampling_ratio=0,
        )
        train_interactions = InteractionDataset(
            train_df, user2idx, item2idx, n_items, config.negative_sampling_ratio, seed=seed,
        )
        val_interactions = InteractionDataset(
            val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=seed,
        )
        trainer = DeepMFTrainer(len(user2idx), n_items, config)
        t0 = time.time()
        trainer.fit(train_interactions, val_interactions)
        val_preds = trainer.predict(val_df, user2idx, item2idx, rating_scale)
        rmse, mae = compute_rmse_mae(val_df["stars"].values, val_preds)
        elapsed = time.time() - t0
        logger.info(
            "[DeepMF] %-20s val RMSE=%.4f MAE=%.4f (%.1fs)", cand["name"], rmse, mae, elapsed,
        )
        results.append({**cand, "val_rmse": float(rmse), "val_mae": float(mae), "seconds": elapsed})
    return results


def tune_cbf(full_df_for_items, train_df, val_df, method, seed, rating_scale) -> list[dict]:
    # sentiment_score dummy KONSTAN -- CBF default include_sentiment=False
    # jadi kolom ini secara struktural wajib ada (build_item_dataframe()
    # selalu agregasi sentiment_agg) tapi TIDAK dipakai sbg fitur, TIDAK
    # perlu skor BERT/keyword sungguhan.
    train_df = train_df.copy()
    train_df["sentiment_score"] = 0.5

    results = []
    for pca_components in CBF_PCA_CANDIDATES:
        cbf_config = CBFConfig(
            method=method, k_min=2, k_max=20, pca_components=pca_components,
            random_state=seed, include_sentiment=False,
        )
        cbf_predictor = CBFPredictor(cbf_config=cbf_config)
        t0 = time.time()
        cbf_predictor.fit(full_df_for_items, train_df)
        val_preds = cbf_predictor.predict(val_df, rating_scale)
        rmse, mae = compute_rmse_mae(val_df["stars"].values, val_preds)
        elapsed = time.time() - t0
        logger.info(
            "[CBF] pca_components=%-4d val RMSE=%.4f MAE=%.4f (K=%d, %.1fs)",
            pca_components, rmse, mae, cbf_predictor.clusterer.best_k, elapsed,
        )
        results.append({
            "pca_components": pca_components, "val_rmse": float(rmse), "val_mae": float(mae),
            "best_k": cbf_predictor.clusterer.best_k, "seconds": elapsed,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/tripadvisor_hotel_config.yaml")
    parser.add_argument(
        "--skip-deepmf", action="store_true",
        help="Lewati sweep DeepMF (mahal, ~2-3 menit/kandidat via single-fit -- TAPI "
        "CATATAN: single-fit val RMSE TERBUKTI TIDAK RELIABLE sbg proxy performa "
        "pipeline sungguhan yg pakai OOF, lihat Temuan 10 -- pakai flag ini utk "
        "sweep CBF SAJA yg murah & valid, DeepMF ditala terpisah lewat regime OOF "
        "yg benar di scripts/tune_deepmf_oof_val.py.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    split_cfg = config["split"]
    seed = exp_cfg["seed"]
    rating_scale = (1.0, 5.0)

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("=== Memuat split (WAJIB sudah ada, load-only) ===")
    splits = UserBasedSplitGenerator.load(Path(split_cfg["output_dir"]))
    train_df, val_df = splits["train"], splits["val"]
    logger.info(
        "Pilot tuning HANYA pakai train (%d baris) + val (%d baris) -- test_df "
        "TIDAK dimuat sama sekali di script ini.", len(train_df), len(val_df),
    )

    logger.info("=== Preprocessing teks (train+val saja) ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)

    all_users = pd.concat([train_df["user_id"], val_df["user_id"]]).unique()
    all_items = pd.concat([train_df["business_id"], val_df["business_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}

    if args.skip_deepmf:
        logger.info("=== Sweep DeepMF DILEWATI (--skip-deepmf) ===")
        deepmf_results = []
        best_deepmf = None
    else:
        logger.info("=== Tuning DeepMF (%d kombinasi) ===", len(DEEPMF_CANDIDATES))
        deepmf_results = tune_deepmf(train_df, val_df, user2idx, item2idx, len(all_items), seed, rating_scale)
        best_deepmf = min(deepmf_results, key=lambda r: r["val_rmse"])

    logger.info("=== Tuning CBF (%d kombinasi) ===", len(CBF_PCA_CANDIDATES))
    full_df_for_items = pd.concat([train_df, val_df], ignore_index=True)
    cbf_results = tune_cbf(
        full_df_for_items, train_df, val_df, config["cbf_clustering"]["method"], seed, rating_scale,
    )
    best_cbf = min(cbf_results, key=lambda r: r["val_rmse"])

    logger.info("=" * 60)
    logger.info("HASIL PILOT TUNING (domain: %s, seed: %d)", exp_cfg["domain"], seed)
    if best_deepmf is not None:
        logger.info("DeepMF terbaik: %s (val RMSE=%.4f)", best_deepmf["name"], best_deepmf["val_rmse"])
    logger.info("CBF terbaik   : pca_components=%d (val RMSE=%.4f)", best_cbf["pca_components"], best_cbf["val_rmse"])
    logger.info("=" * 60)

    output = {
        "domain": exp_cfg["domain"],
        "seed": seed,
        "deepmf_candidates": deepmf_results,
        "deepmf_best": best_deepmf,
        "cbf_candidates": cbf_results,
        "cbf_best": best_cbf,
        "note": (
            "Pilot 1-domain, val set dipakai utk early-stopping DAN model "
            "selection (bukan dev-set terpisah) -- lihat docstring script "
            "ini. test_df TIDAK PERNAH dimuat."
        ),
    }
    output_dir = Path(config["logging"]["checkpoint_dir"])
    output_path = output_dir / "tuning_pilot_deepmf_cbf.yaml"
    output_path.write_text(yaml.dump(output, sort_keys=False), encoding="utf-8")
    logger.info("Hasil pilot disimpan ke %s", output_path)


if __name__ == "__main__":
    main()
