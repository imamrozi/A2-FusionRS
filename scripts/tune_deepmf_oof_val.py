"""
scripts/tune_deepmf_oof_val.py

Tuning DeepMF yang BENAR -- riwayat perbaikan bertahap (lihat memori sesi
utk detail lengkap tiap temuan):

1. (Temuan 10) val RMSE dari single-fit sederhana TIDAK RELIABLE sbg proxy
   performa pipeline sungguhan -- diperbaiki dgn regime OOF+LOO yg sama
   persis dgn deployment.
2. (Temuan 13) SEKALIPUN regime OOF+LOO benar, pencarian 24-kandidat
   SEKALIGUS (coordinate search tanpa jeda verifikasi) tetap gagal --
   config pemenang di val RMSE-nya JUSTRU LEBIH BURUK di test. Diperbaiki
   dgn eksekusi bertahap per-stage + verifikasi test wajib di antaranya.
3. (Temuan B4 audit metodologi) coordinate search py bug: pemenang stage
   tdk dibandingkan ke basis dari stage sebelumnya. Diperbaiki dgn
   `--base-rmse` + perbandingan eksplisit.
4. **(Temuan A2 audit metodologi, INI YANG DIPERBAIKI DI VERSI INI, PALING
   MENDASAR)**: SEKALIPUN 1-3 sudah benar, ternyata val_df dipakai GANDA --
   `DeepMFTrainer.fit()` melakukan early-stopping BERDASARKAN val (simpan/
   restore bobot dgn val RMSE terbaik), LALU model yg SUDAH "mengintip"
   val itu DIEVALUASI LAGI di val yg SAMA utk memilih kandidat. Ini bias
   optimistik sistematis yg BESARNYA BEDA antar kandidat -- penjelasan
   MEKANISTIS kenapa val & test 3x berturut-turut berlawanan arah (Temuan
   13, 16, 18), bukan sekadar "val kecil/noisy".

FIX A2 (perubahan utama versi ini): PISAHKAN peran early-stopping vs
seleksi kandidat ke DUA set yang independen, TANPA menyentuh file split
bersama (train/val/test tetap dipakai apa adanya oleh SEMUA script lain):

  - `train_fit` (~85% dari train_df asli): dipakai fit CBF, fit DeepMF
    (OOF + model penuh) -- pengganti peran "train" utk pencarian ini SAJA.
  - `val_df` (split asli, TIDAK diubah): PERANNYA DIPERSEMPIT -- HANYA
    dipakai early-stopping DeepMF (spt semula), TIDAK PERNAH lagi dipakai
    menilai/membandingkan kandidat.
  - `selection_dev` (~15% dari train_df asli, split SEKALI di awal,
    deterministik thd seed): set BARU, independen dari early-stopping DAN
    dari fitting -- CBF & DeepMF (OOF maupun model penuh) TIDAK PERNAH
    melihat baris ini saat fit. Berperan sbg "test palsu" murni utk
    seleksi kandidat. Predict di sini SELALU pakai method genuinely-out-
    of-sample (`predict()` biasa, BUKAN `predict_train_loo()`/OOF) --
    analog persis test_df di pipeline sungguhan.

test_df ASLI tetap TIDAK PERNAH dimuat sama sekali di script ini -- WAJIB
diverifikasi SEKALI lewat run_baseline_absa.py --results-tag SETELAH
pencarian ini selesai, spt sebelumnya.

Protokol tiap kandidat (biaya ~sama dgn 1 run pipeline penuh):
1. compute_oof_predictions(train_fit, ...) -> train_deepmf_preds (5-fold OOF,
   early-stopping tiap fold pakai val_df).
2. DeepMFTrainer baru, fit(train_fit, val_df utk early-stop) ->
   predict(selection_dev) -> dev_deepmf_preds.
3. CBF: predict_train_loo(train_fit) -> train_cbf_preds ; predict(selection_dev)
   -> dev_cbf_preds (CBF difit SEKALI di awal, pakai train_fit+val_df SAJA
   utk item universe -- selection_dev TIDAK PERNAH masuk fit CBF).
4. sentiment: kolom konstan nol (protokol no_sentiment_ablation).
5. Fusion: fit di (train_deepmf_preds, train_cbf_preds) -> predict di
   (dev_deepmf_preds, dev_cbf_preds) -> dev_fusion_rmse. INI metrik
   seleksi kandidat -- val_df TIDAK MUNCUL SAMA SEKALI di metrik ini lagi.

RESUMABLE: checkpoint CSV v3 (kolom: stage, params_json, dev_fusion_rmse,
seconds) -- NAMA BEDA dari checkpoint v2 (skema lama msh pakai val_fusion_
rmse yg tercemar Temuan A2, TIDAK backward compatible & TIDAK VALID utk
dipercaya lagi -- lihat memori sesi).

Usage (Colab, GPU disarankan):
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage0_optimizer_lr

    # Stage lanjutan, isi --base-rmse dgn dev_fusion_rmse checkpoint stage sebelumnya:
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage1_embedding_dim --base-optimizer adamw --base-learning-rate 0.002 --base-rmse <isi>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.a2fusionrs.selection_split import SELECTION_DEV_FRACTION, split_train_fit_dev
from src.baseline.cbf_clustering import CBFConfig, CBFPredictor
from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset, compute_oof_predictions
from src.baseline.fusion_nmf_dt import FusionConfig, NMFDecisionTreeFusion
from src.config_utils import load_config
from src.evaluation.metrics import compute_rmse_mae
from src.preprocessing import TextPreprocessor
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RATING_SCALE = (1.0, 5.0)
# SELECTION_DEV_FRACTION & split_train_fit_dev() dipindah ke
# src/a2fusionrs/selection_split.py (di-import di atas) supaya bisa dipakai
# ulang oleh run_attention_gated_fusion.py -- perilaku TIDAK berubah.

DEFAULTS = {
    "optimizer": "adamw",
    "learning_rate": 0.002,
    "embedding_dim": 128,
    "hidden_layers": (256, 128, 64, 32),
    "dropout": 0.3,
    "epochs": 20,
    "weight_decay": 0.0,
}

STAGE0_OPTIMIZER_LR: list[dict] = [
    {"optimizer": "sgd", "learning_rate": 0.001},
    {"optimizer": "sgd", "learning_rate": 0.003},
    {"optimizer": "adamw", "learning_rate": 0.0005},
    {"optimizer": "adamw", "learning_rate": 0.001},
    {"optimizer": "adamw", "learning_rate": 0.002},
    {"optimizer": "adamw", "learning_rate": 0.005},
]

ALL_STAGES: dict[str, list[dict]] = {
    "stage0_optimizer_lr": STAGE0_OPTIMIZER_LR,
    "stage1_embedding_dim": [{"embedding_dim": v} for v in [32, 64, 96, 128, 192, 256]],
    "stage2_dropout": [{"dropout": v} for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    "stage3_epochs": [{"epochs": v} for v in [5, 10, 15, 20, 30]],
    "stage4_weight_decay": [{"weight_decay": v} for v in [0.0001, 0.001, 0.01, 0.05]],
}


def load_checkpoint(csv_path: Path) -> dict[tuple[str, str], float]:
    if not csv_path.exists():
        return {}
    done = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[(row["stage"], row["params_json"])] = float(row["dev_fusion_rmse"])
    return done


def append_checkpoint(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def evaluate_candidate(
    params: dict,
    train_fit: pd.DataFrame, val_df: pd.DataFrame, selection_dev: pd.DataFrame,
    user2idx: dict, item2idx: dict, n_items: int,
    train_cbf_preds: np.ndarray, dev_cbf_preds: np.ndarray,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    config = DeepMFConfig(
        embedding_dim=params["embedding_dim"],
        hidden_layers=tuple(params["hidden_layers"]),
        dropout=params["dropout"],
        batch_size=512,
        learning_rate=params["learning_rate"],
        epochs=params["epochs"],
        negative_sampling_ratio=0,
        optimizer=params["optimizer"],
        weight_decay=params["weight_decay"],
    )
    # val_df HANYA utk early-stopping (Fix Temuan A2) -- tidak pernah
    # dipakai lagi sbg target evaluasi kandidat di bawah.
    val_interactions = InteractionDataset(val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=seed)

    train_deepmf_preds = compute_oof_predictions(
        train_fit, val_interactions, user2idx, item2idx, n_items, config, RATING_SCALE, seed=seed,
    )

    torch.manual_seed(seed)
    train_interactions = InteractionDataset(
        train_fit, user2idx, item2idx, n_items, config.negative_sampling_ratio, seed=seed,
    )
    trainer = DeepMFTrainer(len(user2idx), n_items, config)
    trainer.fit(train_interactions, val_interactions)  # val_df early-stop DI SINI SAJA
    dev_deepmf_preds = trainer.predict(selection_dev, user2idx, item2idx, RATING_SCALE)

    train_sentiment = np.zeros(len(train_fit), dtype=np.float32)
    dev_sentiment = np.zeros(len(selection_dev), dtype=np.float32)

    fusion = NMFDecisionTreeFusion(FusionConfig(nmf_components=3, dt_max_depth=10, random_state=seed))
    fusion.fit(
        sentiment_scores=train_sentiment, deepmf_preds=train_deepmf_preds,
        cbf_preds=train_cbf_preds, y_true_ratings=train_fit["stars"].values,
    )
    dev_final_preds = fusion.predict(
        sentiment_scores=dev_sentiment, deepmf_preds=dev_deepmf_preds, cbf_preds=dev_cbf_preds,
    )
    dev_final_preds = np.clip(dev_final_preds, RATING_SCALE[0], RATING_SCALE[1])

    rmse, _ = compute_rmse_mae(selection_dev["stars"].values, dev_final_preds)
    return float(rmse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/tripadvisor_hotel_config.yaml")
    parser.add_argument(
        "--cbf-pca-components", type=int, default=50,
        help="pca_components CBF (ditala TERPISAH via tune_deepmf_cbf_pilot.py --skip-deepmf).",
    )
    parser.add_argument(
        "--stages", type=str, default="stage0_optimizer_lr",
        help="Daftar stage dipisah koma, dijalankan BERURUTAN. Default HANYA "
        "stage0_optimizer_lr -- WAJIB diverifikasi ke test set dulu sebelum "
        "menjalankan stage lain. Pilihan: " + ", ".join(ALL_STAGES.keys()),
    )
    parser.add_argument("--base-optimizer", type=str, default=None)
    parser.add_argument("--base-learning-rate", type=float, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--base-weight-decay", type=float, default=None)
    parser.add_argument(
        "--base-rmse", type=float, default=None,
        help="dev_fusion_rmse basis SAAT INI dari checkpoint stage sebelumnya -- WAJIB "
        "diisi kalau melanjutkan dari stage lain (bukan run pertama), lihat Temuan B4.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    split_cfg = config["split"]
    seed = exp_cfg["seed"]

    np.random.seed(seed)

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # v3: skema BEDA dari v2 (val_fusion_rmse, tercemar Temuan A2 -- val
    # dipakai ganda utk early-stop DAN seleksi). v3 pakai dev_fusion_rmse
    # dari selection_dev yg independen. TIDAK backward compatible dgn v2.
    csv_path = checkpoint_dir / "tuning_deepmf_oof_dev_search_v3.csv"
    done = load_checkpoint(csv_path)
    logger.info("Checkpoint: %d kandidat sudah selesai sebelumnya (resume dari %s)", len(done), csv_path)

    logger.info("=== Memuat split (WAJIB sudah ada, load-only) -- test_df TIDAK dimuat ===")
    splits = UserBasedSplitGenerator.load(Path(split_cfg["output_dir"]))
    train_df, val_df = splits["train"], splits["val"]

    logger.info("=== Preprocessing teks (train+val saja) ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)

    # Fix Temuan A2: pisah train_df jadi train_fit (fitting) + selection_dev
    # (seleksi kandidat, independen dari early-stopping val_df).
    train_fit, selection_dev = split_train_fit_dev(train_df, seed, SELECTION_DEV_FRACTION)

    all_users = pd.concat([train_df["user_id"], val_df["user_id"]]).unique()
    all_items = pd.concat([train_df["business_id"], val_df["business_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}

    logger.info("=== Fit CBF sekali (pca_components=%d, reuse lintas kandidat DeepMF) ===", args.cbf_pca_components)
    train_fit_cbf = train_fit.copy()
    train_fit_cbf["sentiment_score"] = 0.5
    # full_df_for_items HANYA train_fit + val_df -- selection_dev SENGAJA
    # TIDAK dimasukkan, supaya benar2 unseen oleh CBF (analog persis test_df
    # di pipeline sungguhan, bukan sekadar LOO-corrected train row).
    full_df_for_items = pd.concat([train_fit, val_df], ignore_index=True)
    cbf_config = CBFConfig(
        method=config["cbf_clustering"]["method"], k_min=2, k_max=20,
        pca_components=args.cbf_pca_components, random_state=seed, include_sentiment=False,
    )
    cbf_predictor = CBFPredictor(cbf_config=cbf_config)
    cbf_predictor.fit(full_df_for_items, train_fit_cbf)
    train_cbf_preds = cbf_predictor.predict_train_loo(train_fit, RATING_SCALE)
    # selection_dev: predict() BIASA (bukan LOO) -- baris ini tidak pernah
    # masuk fit CBF sama sekali, genuinely out-of-sample spt test_df asli.
    dev_cbf_preds = cbf_predictor.predict(selection_dev, RATING_SCALE)

    current_best = dict(DEFAULTS)
    if args.base_optimizer is not None:
        current_best["optimizer"] = args.base_optimizer
    if args.base_learning_rate is not None:
        current_best["learning_rate"] = args.base_learning_rate
    if args.base_epochs is not None:
        current_best["epochs"] = args.base_epochs
    if args.base_weight_decay is not None:
        current_best["weight_decay"] = args.base_weight_decay
    current_best_rmse = args.base_rmse
    n_items = len(all_items)

    stage_names = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage_name in stage_names:
        if stage_name not in ALL_STAGES:
            raise ValueError(f"Stage '{stage_name}' tidak dikenal -- pilihan: {list(ALL_STAGES.keys())}")
        candidates = ALL_STAGES[stage_name]
        logger.info(
            "=== %s (%d kandidat) -- basis: %s (dev RMSE basis=%s) ===",
            stage_name, len(candidates), current_best,
            f"{current_best_rmse:.4f}" if current_best_rmse is not None else "BELUM DIKETAHUI",
        )
        stage_results = []
        for override in candidates:
            params = dict(current_best)
            params.update(override)
            params_key = json.dumps(override, sort_keys=True)
            cache_key = (stage_name, params_key)

            if cache_key in done:
                logger.info("[%s] %s SUDAH ADA di checkpoint (dev RMSE=%.4f) -- skip.", stage_name, override, done[cache_key])
                stage_results.append((override, done[cache_key]))
                continue

            t0 = time.time()
            rmse = evaluate_candidate(
                params, train_fit, val_df, selection_dev, user2idx, item2idx, n_items,
                train_cbf_preds, dev_cbf_preds, seed,
            )
            elapsed = time.time() - t0
            logger.info("[%s] %s dev_fusion_RMSE=%.4f (%.1f menit)", stage_name, override, rmse, elapsed / 60)
            append_checkpoint(csv_path, {
                "stage": stage_name, "params_json": params_key,
                "dev_fusion_rmse": rmse, "seconds": elapsed,
            })
            stage_results.append((override, rmse))

        best_override, best_rmse = min(stage_results, key=lambda t: t[1])

        if current_best_rmse is not None and best_rmse >= current_best_rmse:
            logger.warning(
                "=== %s SELESAI: TIDAK ADA kandidat baru yg mengalahkan basis "
                "(terbaik stage ini %s dev RMSE=%.4f, vs basis dev RMSE=%.4f) -- "
                "current_best TIDAK diubah, tetap %s ===",
                stage_name, best_override, best_rmse, current_best_rmse, current_best,
            )
        else:
            current_best.update(best_override)
            current_best_rmse = best_rmse
            logger.info("=== %s SELESAI: terbaik = %s (dev RMSE=%.4f) ===", stage_name, best_override, best_rmse)

    logger.info("=" * 60)
    logger.info("STAGE(S) SELESAI -- config DeepMF terbaik SEJAUH INI:")
    for k, v in current_best.items():
        logger.info("  %s = %s", k, v)
    logger.info(
        "dev_fusion_rmse basis: %s",
        f"{current_best_rmse:.4f}" if current_best_rmse is not None else "BELUM DIKETAHUI",
    )
    logger.info("CBF pca_components dipakai (ditala terpisah): %d", args.cbf_pca_components)
    logger.info("=" * 60)
    logger.info(
        "WAJIB: verifikasi config di atas ke TEST SET SEKALI lewat run_baseline_absa.py "
        "--results-tag <tag> --sentiment-protocol no_sentiment_ablation SEBELUM "
        "mempercayai dev_fusion_rmse di atas ATAU melanjutkan ke stage berikutnya."
    )


if __name__ == "__main__":
    main()
