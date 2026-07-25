"""
scripts/tune_deepmf_oof_val.py

Tuning DeepMF yang BENAR (lihat Temuan 10, memori sesi): val RMSE dari
single-fit sederhana (scripts/tune_deepmf_cbf_pilot.py) TERBUKTI TIDAK
RELIABLE sbg proxy performa pipeline sungguhan -- config yang tampak lebih
baik di situ (learning_rate 0,005) justru RMSE test-nya memburuk 38,6%
lewat pipeline penuh (1,1183 -> 1,5494), krn pipeline sungguhan memakai
`compute_oof_predictions()` (5-fold OOF, ~80% data/fold) utk
`train_deepmf_preds`, BUKAN satu model dilatih full data spt di pilot.

Script ini mensimulasikan REGIME YANG SAMA PERSIS dgn deployment (OOF utk
stream train, model penuh utk stream held-out) TAPI held-out di sini
adalah VAL, BUKAN TEST -- test_df TIDAK PERNAH dimuat sama sekali di
script ini. Ini penting supaya model selection tidak bocor ke test set
(kalau kita pilih kandidat berdasar RMSE test, test berubah fungsi jadi
tuning set -- angka final jadi optimis palsu). Test HANYA disentuh SEKALI
di akhir, lewat run_baseline_absa.py dgn config hasil kandidat pemenang,
utk melaporkan angka final yang jujur.

Protokol tiap kandidat (biaya ~sama dgn 1 run pipeline penuh, ~15-18 menit
lokal CPU -- KEMUNGKINAN lebih cepat di Colab GPU):
1. compute_oof_predictions(train_df, ...) -> train_deepmf_preds (5-fold OOF)
2. DeepMFTrainer baru, fit(train, val) -> predict(val_df) -> val_deepmf_preds
   (model PENUH, analog persis test_deepmf_preds di pipeline sungguhan)
3. CBF: predict_train_loo(train_df) -> train_cbf_preds ; predict(val_df) ->
   val_cbf_preds -- CBF DIFIT SEKALI SAJA (independen dari hyperparameter
   DeepMF), di-reuse lintas SEMUA kandidat DeepMF -- besar penghematan.
4. sentiment: KOLOM KONSTAN nol (protokol no_sentiment_ablation) -- isolasi
   murni kontribusi DeepMF+CBF, konsisten dgn temuan floor sebelumnya
   (RMSE 1,1183 untuk hyperparameter default).
5. Fusion NMF+DT: fit(train_deepmf_preds, train_cbf_preds, sentiment=0,
   y=train_df.stars) -> predict(val_deepmf_preds, val_cbf_preds,
   sentiment=0) -> val_fusion_rmse. INI metrik seleksi kandidat.

PENCARIAN: coordinate/greedy search bertahap (bukan grid penuh -- grid
penuh utk 4 hyperparameter x 5-8 nilai tiap = ratusan kandidat, tidak
feasible). Tiap tahap men-tala SATU hyperparameter, memakai nilai TERBAIK
dari tahap sebelumnya utk hyperparameter lain:
  Tahap 1: learning_rate (default lain: embedding=128, hidden=[256,128,64,32], dropout=0.3, epochs=20)
  Tahap 2: embedding_dim (pakai learning_rate terbaik Tahap 1)
  Tahap 3: dropout (pakai learning_rate+embedding_dim terbaik)
  Tahap 4: epochs (Temuan 9: val RMSE higher_lr blm plateau di epoch 20)

RESUMABLE: tiap kandidat yang sudah selesai ditulis ke CSV checkpoint
(append per baris) -- restart script SKIP kandidat yang sudah ada di CSV
(dicocokkan by (stage, param_name, param_value)), sama pola dgn
scripts/rerun_cbf_nosentiment_full.sh (tahan terputus sesi Colab).

Usage (Colab, GPU disarankan -- lihat DeepMFConfig.device auto-detect cuda):
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULTS = {
    "embedding_dim": 128,
    "hidden_layers": (256, 128, 64, 32),
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 20,
}

# (stage, param_name, candidate_values) -- dieksekusi berurutan, tiap tahap
# pakai nilai TERBAIK tahap sebelumnya utk param lain (coordinate search).
SEARCH_STAGES: list[tuple[str, str, list]] = [
    ("stage1_learning_rate", "learning_rate", [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]),
    ("stage2_embedding_dim", "embedding_dim", [32, 64, 96, 128, 192, 256]),
    ("stage3_dropout", "dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    ("stage4_epochs", "epochs", [20, 30, 40, 50]),
]


def load_checkpoint(csv_path: Path) -> dict[tuple[str, str, str], float]:
    if not csv_path.exists():
        return {}
    done = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["stage"], row["param_name"], row["param_value"])
            done[key] = float(row["val_fusion_rmse"])
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
    train_df, val_df, user2idx, item2idx, n_items,
    train_cbf_preds: np.ndarray, val_cbf_preds: np.ndarray,
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
    )
    val_interactions = InteractionDataset(val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=seed)

    # 1. train_deepmf_preds via OOF 5-fold -- SAMA PERSIS mekanisme run_baseline_absa.py.
    train_deepmf_preds = compute_oof_predictions(
        train_df, val_interactions, user2idx, item2idx, n_items, config, RATING_SCALE, seed=seed,
    )

    # 2. Model PENUH (dilatih full train_df) -> prediksi val_df, analog persis
    #    test_deepmf_preds di pipeline sungguhan (val di sini = pengganti test,
    #    supaya test_df asli tidak pernah tersentuh selama tuning).
    torch.manual_seed(seed)
    train_interactions = InteractionDataset(
        train_df, user2idx, item2idx, n_items, config.negative_sampling_ratio, seed=seed,
    )
    trainer = DeepMFTrainer(len(user2idx), n_items, config)
    trainer.fit(train_interactions, val_interactions)
    val_deepmf_preds = trainer.predict(val_df, user2idx, item2idx, RATING_SCALE)

    # 3. Sentiment: kolom konstan nol (protokol no_sentiment_ablation) --
    #    isolasi murni DeepMF+CBF, konsisten dgn floor yg sudah diukur.
    train_sentiment = np.zeros(len(train_df), dtype=np.float32)
    val_sentiment = np.zeros(len(val_df), dtype=np.float32)

    # 4. Fusion: fit di train (OOF deepmf + LOO cbf), evaluasi di VAL (bukan test).
    fusion = NMFDecisionTreeFusion(FusionConfig(nmf_components=3, dt_max_depth=10, random_state=seed))
    fusion.fit(
        sentiment_scores=train_sentiment, deepmf_preds=train_deepmf_preds,
        cbf_preds=train_cbf_preds, y_true_ratings=train_df["stars"].values,
    )
    val_final_preds = fusion.predict(
        sentiment_scores=val_sentiment, deepmf_preds=val_deepmf_preds, cbf_preds=val_cbf_preds,
    )
    val_final_preds = np.clip(val_final_preds, RATING_SCALE[0], RATING_SCALE[1])

    rmse, _ = compute_rmse_mae(val_df["stars"].values, val_final_preds)
    return float(rmse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/tripadvisor_hotel_config.yaml")
    parser.add_argument(
        "--cbf-pca-components", type=int, default=50,
        help="pca_components CBF (ditala TERPISAH & lebih murah lewat "
        "scripts/tune_deepmf_cbf_pilot.py --skip-deepmf -- isi nilai "
        "pemenang dari situ di sini).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    split_cfg = config["split"]
    seed = exp_cfg["seed"]

    np.random.seed(seed)

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = checkpoint_dir / "tuning_deepmf_oof_val_search.csv"
    done = load_checkpoint(csv_path)
    logger.info("Checkpoint: %d kandidat sudah selesai sebelumnya (resume dari %s)", len(done), csv_path)

    logger.info("=== Memuat split (WAJIB sudah ada, load-only) -- test_df TIDAK dimuat ===")
    splits = UserBasedSplitGenerator.load(Path(split_cfg["output_dir"]))
    train_df, val_df = splits["train"], splits["val"]

    logger.info("=== Preprocessing teks (train+val saja) ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)

    all_users = pd.concat([train_df["user_id"], val_df["user_id"]]).unique()
    all_items = pd.concat([train_df["business_id"], val_df["business_id"]]).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}

    # CBF difit SEKALI (independen dari hyperparameter DeepMF), reuse lintas
    # SEMUA kandidat DeepMF -- sentiment_score dummy konstan (structural only,
    # include_sentiment=False mengabaikannya, lihat pilot script).
    logger.info("=== Fit CBF sekali (pca_components=%d, reuse lintas kandidat DeepMF) ===", args.cbf_pca_components)
    train_df_cbf = train_df.copy()
    train_df_cbf["sentiment_score"] = 0.5
    full_df_for_items = pd.concat([train_df, val_df], ignore_index=True)
    cbf_config = CBFConfig(
        method=config["cbf_clustering"]["method"], k_min=2, k_max=20,
        pca_components=args.cbf_pca_components, random_state=seed, include_sentiment=False,
    )
    cbf_predictor = CBFPredictor(cbf_config=cbf_config)
    cbf_predictor.fit(full_df_for_items, train_df_cbf)
    train_cbf_preds = cbf_predictor.predict_train_loo(train_df, RATING_SCALE)
    val_cbf_preds = cbf_predictor.predict(val_df, RATING_SCALE)

    current_best = dict(DEFAULTS)
    n_items = len(all_items)

    for stage_name, param_name, candidates in SEARCH_STAGES:
        logger.info("=== %s: tala '%s' (%d kandidat) ===", stage_name, param_name, len(candidates))
        stage_results = []
        for value in candidates:
            key = (stage_name, param_name, str(value))
            if key in done:
                logger.info("[%s] %s=%s SUDAH ADA di checkpoint (val RMSE=%.4f) -- skip.", stage_name, param_name, value, done[key])
                stage_results.append((value, done[key]))
                continue

            params = dict(current_best)
            params[param_name] = value
            t0 = time.time()
            rmse = evaluate_candidate(params, train_df, val_df, user2idx, item2idx, n_items, train_cbf_preds, val_cbf_preds, seed)
            elapsed = time.time() - t0
            logger.info("[%s] %s=%-8s val_fusion_RMSE=%.4f (%.1f menit)", stage_name, param_name, value, rmse, elapsed / 60)
            append_checkpoint(csv_path, {
                "stage": stage_name, "param_name": param_name, "param_value": str(value),
                "val_fusion_rmse": rmse, "seconds": elapsed,
            })
            stage_results.append((value, rmse))

        best_value, best_rmse = min(stage_results, key=lambda t: t[1])
        current_best[param_name] = best_value
        logger.info("=== %s SELESAI: %s terbaik = %s (val RMSE=%.4f) ===", stage_name, param_name, best_value, best_rmse)

    logger.info("=" * 60)
    logger.info("PENCARIAN SELESAI -- config DeepMF terbaik ditemukan:")
    for k, v in current_best.items():
        logger.info("  %s = %s", k, v)
    logger.info("CBF pca_components dipakai (ditala terpisah): %d", args.cbf_pca_components)
    logger.info("=" * 60)
    logger.info(
        "LANGKAH TERAKHIR (WAJIB, test_df belum pernah disentuh sama sekali): "
        "buat config YAML dgn nilai di atas, jalankan run_baseline_absa.py SEKALI "
        "(no_sentiment_ablation dulu utk konfirmasi floor, lalu target_review) utk "
        "dapat angka test RMSE final yang jujur."
    )


if __name__ == "__main__":
    main()
