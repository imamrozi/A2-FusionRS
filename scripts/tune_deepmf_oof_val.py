"""
scripts/tune_deepmf_oof_val.py

Tuning DeepMF yang BENAR (lihat Temuan 10, memori sesi): val RMSE dari
single-fit sederhana (scripts/tune_deepmf_cbf_pilot.py) TERBUKTI TIDAK
RELIABLE sbg proxy performa pipeline sungguhan -- config yang tampak lebih
baik di situ (learning_rate 0,005) justru RMSE test-nya memburuk 38,6%
lewat pipeline penuh, krn pipeline sungguhan memakai
`compute_oof_predictions()` (5-fold OOF, ~80% data/fold) utk
`train_deepmf_preds`, BUKAN satu model dilatih full data spt di pilot.

Script ini mensimulasikan REGIME YANG SAMA PERSIS dgn deployment (OOF utk
stream train, model penuh utk stream held-out) TAPI held-out di sini
adalah VAL, BUKAN TEST -- test_df TIDAK PERNAH dimuat sama sekali di
script ini.

PERINGATAN KERAS (Temuan 13, lihat memori sesi): SEKALIPUN regime OOF+LOO
sudah benar & val dipakai (bukan test), pencarian 24-kandidat PERTAMA
(coordinate search murni learning_rate/embedding_dim/dropout/epochs)
TETAP GAGAL -- config pemenang di val (RMSE 0,9317) justru RMSE TEST-nya
LEBIH BURUK dari default (1,3065 vs 1,1183). Root cause diduga: val set
kecil (3.487 baris) + 24 percobaan pada metrik yg noisy (variansi run-to-
run besar) = val-set overfitting/multiple-comparisons, BUKAN perbaikan
general. KONSEKUENSI DESAIN: script ini SEKARANG wajib dijalankan
BERTAHAP per stage (--stages), BUKAN semua stage sekaligus tanpa jeda --
verifikasi ke test SETELAH stage optimizer/lr selesai, SEBELUM lanjut ke
stage embedding_dim/dropout/epochs, supaya tidak menghabiskan budget
komputasi memperluas config yang ternyata ilusi.

STAGE 0 (BARU, axis yang belum pernah dicoba): optimizer. DeepMFTrainer
SEBELUMNYA hardcode SGD polos (tanpa momentum/weight_decay) -- src/
baseline/deepmf.py baris ~154, terbukti SANGAT sensitif thd learning_rate
(band stabil sempit, kolaps total ke prediktor konstan di lr yg naik
sedikit -- lr=0,003 val RMSE 0,93 -> lr=0,005 val RMSE 3,07). Adam/AdamW
kini didukung (DeepMFConfig.optimizer). Stage 0 mencoba optimizer x
learning_rate SEKALIGUS (bukan terpisah spt stage lr lama) krn dua-duanya
diketahui berinteraksi kuat.

Protokol tiap kandidat (biaya ~sama dgn 1 run pipeline penuh, ~15-18 menit
lokal CPU):
1. compute_oof_predictions(train_df, ...) -> train_deepmf_preds (5-fold OOF)
2. DeepMFTrainer baru, fit(train, val) -> predict(val_df) -> val_deepmf_preds
   (model PENUH, analog persis test_deepmf_preds di pipeline sungguhan)
3. CBF: predict_train_loo(train_df) -> train_cbf_preds ; predict(val_df) ->
   val_cbf_preds -- CBF DIFIT SEKALI SAJA (independen dari hyperparameter
   DeepMF), di-reuse lintas SEMUA kandidat DeepMF.
4. sentiment: KOLOM KONSTAN nol (protokol no_sentiment_ablation) -- isolasi
   murni kontribusi DeepMF+CBF.
5. Fusion NMF+DT: fit(train_deepmf_preds, train_cbf_preds, sentiment=0,
   y=train_df.stars) -> predict(val_deepmf_preds, val_cbf_preds,
   sentiment=0) -> val_fusion_rmse. Metrik seleksi kandidat DI DALAM 1
   stage -- TAPI keputusan LANJUT/TIDAK ke stage berikutnya WAJIB
   diverifikasi dulu ke test set via run_baseline_absa.py --results-tag,
   BUKAN otomatis dipercaya dari val_fusion_rmse saja (lihat peringatan
   di atas).

RESUMABLE: checkpoint CSV (kolom: stage, params_json, val_fusion_rmse,
seconds) -- restart skip kandidat yang params_json-nya sudah ada persis
sama di CSV untuk stage yang sama.

PENTING (Temuan B4 audit metodologi, reports/methodology_audit_2026-07-26.md):
SETIAP stage LANJUTAN (bukan stage pertama dlm satu invocation) WAJIB diisi
`--base-rmse` dgn val_fusion_rmse dari config basis SAAT INI -- TANPA ini,
script cuma bandingkan sesama kandidat BARU dlm satu stage, tidak pernah
cross-check ke basis dari stage sebelumnya, dan bisa "memenangkan" kandidat
yg sebenarnya LEBIH BURUK dari basis (ini PERSIS yg terjadi historisnya di
stage_adamw_epochs sebelum fix ini ada).

Usage (Colab, GPU disarankan):
    # Stage 0 SAJA dulu (optimizer x lr) -- run pertama, tidak ada basis
    # diketahui, --base-rmse tidak perlu diisi. WAJIB diverifikasi ke test
    # sebelum lanjut stage lain.
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage0_optimizer_lr

    # Stage lanjutan (embedding_dim/dropout/epochs), pakai optimizer+lr
    # pemenang stage 0 sbg basis -- override manual via --base-optimizer/--base-lr
    # DAN --base-rmse (val_fusion_rmse config itu dari checkpoint stage0).
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage1_embedding_dim,stage2_dropout,stage3_epochs --base-optimizer adam --base-learning-rate 0.001 --base-rmse 0.9803

    # Stage AdamW lanjutan (contoh, isi --base-rmse dgn angka checkpoint yg
    # sesuai basis yg dipakai):
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage_adamw_epochs --base-optimizer adamw --base-learning-rate 0.002 --base-rmse 0.9803 --cbf-pca-components 90
    # setelah verifikasi test stage_adamw_epochs, lanjut (isi --base-epochs manual dgn pemenang):
    python scripts/tune_deepmf_oof_val.py --config configs/tripadvisor_hotel_config.yaml --stages stage_adamw_weight_decay --base-optimizer adamw --base-learning-rate 0.002 --base-epochs <pemenang> --cbf-pca-components 90
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
    "optimizer": "sgd",
    "learning_rate": 0.001,
    "embedding_dim": 128,
    "hidden_layers": (256, 128, 64, 32),
    "dropout": 0.3,
    "epochs": 20,
    "weight_decay": 0.0,
}

# Stage 0: optimizer x learning_rate SEKALIGUS (interaksi kuat, tidak bisa
# dipisah spt coordinate search murni). 2 anchor SGD (default + "pemenang"
# lama yg TERBUKTI gagal di test, Temuan 13 -- disertakan lagi di sini
# sbg pembanding langsung dlm skema checkpoint baru) + Adam/AdamW di
# beberapa lr yg relevan (skala umum Adam-family, 5e-4 s/d 5e-3).
STAGE0_OPTIMIZER_LR: list[dict] = [
    {"optimizer": "sgd", "learning_rate": 0.001},
    {"optimizer": "sgd", "learning_rate": 0.003},
    {"optimizer": "adam", "learning_rate": 0.0005},
    {"optimizer": "adam", "learning_rate": 0.001},
    {"optimizer": "adam", "learning_rate": 0.002},
    {"optimizer": "adam", "learning_rate": 0.005},
    {"optimizer": "adamw", "learning_rate": 0.0005},
    {"optimizer": "adamw", "learning_rate": 0.001},
    {"optimizer": "adamw", "learning_rate": 0.002},
    {"optimizer": "adamw", "learning_rate": 0.005},
]

ALL_STAGES: dict[str, list[dict]] = {
    "stage0_optimizer_lr": STAGE0_OPTIMIZER_LR,
    "stage1_embedding_dim": [{"embedding_dim": v} for v in [32, 64, 96, 128, 192, 256]],
    "stage2_dropout": [{"dropout": v} for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    "stage3_epochs": [{"epochs": v} for v in [20, 30, 40, 50]],
    # Stage khusus AdamW (Temuan 16, memori sesi): verifikasi test Adam
    # lr=0,002/epochs=20 near-miss (test RMSE 1,1309 vs default 1,1183,
    # +1,1%) -- log training tunjukkan train MSE turun sampai 0,002 sementara
    # val RMSE BEROSILASI (tidak monoton lg) -- diagnosis overfitting jelas,
    # epochs=20 (diwarisi era tuning SGD) kemungkinan kelewat banyak utk
    # Adam yg konvergen jauh lbh cepat. 20 SENGAJA tidak diulang di sini
    # (sudah py hasil dari stage0). Jalankan dgn --base-optimizer adamw
    # --base-learning-rate 0.002 SETELAH stage_adamw_epochs selesai &
    # diverifikasi ke test, baru lanjut stage_adamw_weight_decay (0,0
    # SENGAJA tdk diulang -- itu PERSIS Adam lr=0,002 yg sudah py hasil,
    # weight_decay=0 membuat AdamW == Adam scr matematis, lihat Temuan 15).
    "stage_adamw_epochs": [{"epochs": v} for v in [3, 5, 8, 12]],
    "stage_adamw_weight_decay": [{"weight_decay": v} for v in [0.0001, 0.001, 0.01, 0.05]],
}


def load_checkpoint(csv_path: Path) -> dict[tuple[str, str], float]:
    if not csv_path.exists():
        return {}
    done = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[(row["stage"], row["params_json"])] = float(row["val_fusion_rmse"])
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
        optimizer=params["optimizer"],
        weight_decay=params["weight_decay"],
    )
    val_interactions = InteractionDataset(val_df, user2idx, item2idx, n_items, negative_ratio=0, seed=seed)

    train_deepmf_preds = compute_oof_predictions(
        train_df, val_interactions, user2idx, item2idx, n_items, config, RATING_SCALE, seed=seed,
    )

    torch.manual_seed(seed)
    train_interactions = InteractionDataset(
        train_df, user2idx, item2idx, n_items, config.negative_sampling_ratio, seed=seed,
    )
    trainer = DeepMFTrainer(len(user2idx), n_items, config)
    trainer.fit(train_interactions, val_interactions)
    val_deepmf_preds = trainer.predict(val_df, user2idx, item2idx, RATING_SCALE)

    train_sentiment = np.zeros(len(train_df), dtype=np.float32)
    val_sentiment = np.zeros(len(val_df), dtype=np.float32)

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
        help="pca_components CBF (ditala TERPISAH via tune_deepmf_cbf_pilot.py --skip-deepmf).",
    )
    parser.add_argument(
        "--stages", type=str, default="stage0_optimizer_lr",
        help="Daftar stage dipisah koma, dijalankan BERURUTAN sesuai urutan disebut "
        "(bukan urutan ALL_STAGES). Default HANYA stage0_optimizer_lr -- WAJIB "
        "diverifikasi ke test set dulu (run_baseline_absa.py --results-tag) sebelum "
        "menjalankan stage lain, lihat peringatan di docstring. Pilihan: " +
        ", ".join(ALL_STAGES.keys()),
    )
    parser.add_argument(
        "--base-optimizer", type=str, default=None,
        help="Override optimizer basis utk stage1-3 (default: DEFAULTS['optimizer'] "
        "kalau tidak diisi -- isi manual dgn pemenang stage0 yg SUDAH diverifikasi ke test).",
    )
    parser.add_argument("--base-learning-rate", type=float, default=None)
    parser.add_argument(
        "--base-epochs", type=int, default=None,
        help="Override epochs basis (mis. utk stage_adamw_weight_decay, isi dgn "
        "pemenang stage_adamw_epochs yg sudah diverifikasi ke test).",
    )
    parser.add_argument(
        "--base-weight-decay", type=float, default=None,
        help="Override weight_decay basis.",
    )
    parser.add_argument(
        "--base-rmse", type=float, default=None,
        help="val_fusion_rmse basis SAAT INI (mis. isi dgn val_fusion_rmse dari config "
        "--base-optimizer/--base-learning-rate/dst di atas, kalau sudah pernah diukur di "
        "stage/run sebelumnya) -- WAJIB diisi kalau melanjutkan dari stage lain (bukan "
        "run pertama), lihat Temuan B4 audit metodologi (reports/methodology_audit_"
        "2026-07-26.md): TANPA ini, script cuma bandingkan sesama kandidat BARU dlm satu "
        "stage, TIDAK PERNAH cross-check ke baseline yg dibawa dari stage sebelumnya --"
        "bisa 'memenangkan' kandidat yg sebenarnya LEBIH BURUK dari basis (persis yg "
        "terjadi di stage_adamw_epochs: epochs=5 dilaporkan 'menang' padahal epochs=20 "
        "basis msh lbh baik). Kalau kosong (run pertama, tidak py basis diketahui), stage "
        "PERTAMA diterima apa adanya (tidak ada yg dibandingkan).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    split_cfg = config["split"]
    seed = exp_cfg["seed"]

    np.random.seed(seed)

    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # NAMA BEDA dari checkpoint pencarian LAMA (tuning_deepmf_oof_val_search.csv,
    # skema kolom param_name/param_value, coordinate search 24-kandidat yg
    # gagal di test -- Temuan 12/13) -- skema checkpoint script ini beda
    # (params_json), TIDAK backward compatible, sengaja file terpisah supaya
    # riwayat pencarian lama tetap ada & tidak collision/corrupt.
    csv_path = checkpoint_dir / "tuning_deepmf_oof_val_search_v2.csv"
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
    if args.base_optimizer is not None:
        current_best["optimizer"] = args.base_optimizer
    if args.base_learning_rate is not None:
        current_best["learning_rate"] = args.base_learning_rate
    if args.base_epochs is not None:
        current_best["epochs"] = args.base_epochs
    if args.base_weight_decay is not None:
        current_best["weight_decay"] = args.base_weight_decay
    current_best_rmse = args.base_rmse  # None -> stage pertama, tidak ada basis diketahui
    n_items = len(all_items)

    stage_names = [s.strip() for s in args.stages.split(",") if s.strip()]
    for stage_name in stage_names:
        if stage_name not in ALL_STAGES:
            raise ValueError(f"Stage '{stage_name}' tidak dikenal -- pilihan: {list(ALL_STAGES.keys())}")
        candidates = ALL_STAGES[stage_name]
        logger.info(
            "=== %s (%d kandidat) -- basis: %s (val RMSE basis=%s) ===",
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
                logger.info("[%s] %s SUDAH ADA di checkpoint (val RMSE=%.4f) -- skip.", stage_name, override, done[cache_key])
                stage_results.append((override, done[cache_key]))
                continue

            t0 = time.time()
            rmse = evaluate_candidate(params, train_df, val_df, user2idx, item2idx, n_items, train_cbf_preds, val_cbf_preds, seed)
            elapsed = time.time() - t0
            logger.info("[%s] %s val_fusion_RMSE=%.4f (%.1f menit)", stage_name, override, rmse, elapsed / 60)
            append_checkpoint(csv_path, {
                "stage": stage_name, "params_json": params_key,
                "val_fusion_rmse": rmse, "seconds": elapsed,
            })
            stage_results.append((override, rmse))

        best_override, best_rmse = min(stage_results, key=lambda t: t[1])

        # FIX Temuan B4 (audit metodologi): bandingkan pemenang stage ini ke
        # BASIS yg dibawa dari stage/run sebelumnya (current_best_rmse), BUKAN
        # cuma sesama kandidat baru dlm stage ini. Kalau tdk ada kandidat baru
        # yg lebih baik dari basis, current_best TIDAK berubah -- stage ini
        # dianggap "tidak menemukan perbaikan", bukan diam-diam ganti ke
        # kandidat yg sebenarnya lebih buruk.
        if current_best_rmse is not None and best_rmse >= current_best_rmse:
            logger.warning(
                "=== %s SELESAI: TIDAK ADA kandidat baru yg mengalahkan basis "
                "(terbaik stage ini %s val RMSE=%.4f, vs basis val RMSE=%.4f) -- "
                "current_best TIDAK diubah, tetap %s ===",
                stage_name, best_override, best_rmse, current_best_rmse, current_best,
            )
        else:
            current_best.update(best_override)
            current_best_rmse = best_rmse
            logger.info("=== %s SELESAI: terbaik = %s (val RMSE=%.4f) ===", stage_name, best_override, best_rmse)

    logger.info("=" * 60)
    logger.info("STAGE(S) SELESAI -- config DeepMF terbaik SEJAUH INI:")
    for k, v in current_best.items():
        logger.info("  %s = %s", k, v)
    logger.info(
        "val_fusion_rmse basis: %s",
        f"{current_best_rmse:.4f}" if current_best_rmse is not None else "BELUM DIKETAHUI",
    )
    logger.info("CBF pca_components dipakai (ditala terpisah): %d", args.cbf_pca_components)
    logger.info("=" * 60)
    logger.info(
        "WAJIB (lihat peringatan Temuan 13 di docstring): verifikasi config di atas "
        "ke TEST SET SEKALI lewat run_baseline_absa.py --results-tag <tag> "
        "--sentiment-protocol no_sentiment_ablation SEBELUM mempercayai val_fusion_rmse "
        "di atas ATAU melanjutkan ke stage berikutnya. Val set kecil (3.487 baris di "
        "domain hotel) rawan overfitting kalau banyak kandidat dicoba."
    )


if __name__ == "__main__":
    main()
