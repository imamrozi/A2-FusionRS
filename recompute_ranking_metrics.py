"""
recompute_ranking_metrics.py

Utilitas SEKALI-PAKAI: recompute precision/recall/NDCG@K dari file
predictions_*.csv yang sudah ada di --results-dir, TANPA training ulang
(DeepMF/CBF/BERT semua di-skip -- cukup baca prediksi per-sampel yang sudah
tersimpan). Dibuat khusus utk migrasi setelah bugfix di
precision_recall_ndcg_at_k() (src/evaluation/metrics.py) -- versi lama bisa
menghasilkan recall/NDCG > 1.0 kalau user punya >1 baris test utk item yang
sama (duplikat user-item di test set tidak di-dedup sebelum dihitung hit).
rmse/mae/precision_at_k TIDAK berubah oleh bugfix ini (dibiarkan apa adanya),
cuma recall_at_k dan ndcg_at_k yang ditimpa dengan nilai yang benar.

Pemakaian:
    python recompute_ranking_metrics.py --results-dir checkpoints/results
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.metrics import precision_recall_ndcg_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 4.0  # sama persis dgn run_baseline.py/run_baseline_absa.py/run_classical_cf.py


def recompute_one(pred_csv: Path, results_dir: Path) -> None:
    yaml_name = pred_csv.name.replace("predictions_", "", 1).replace(".csv", ".yaml")
    yaml_path = results_dir / yaml_name
    if not yaml_path.exists():
        logger.warning("Lewati %s: file hasil %s tidak ditemukan.", pred_csv.name, yaml_path)
        return

    with open(yaml_path) as f:
        results = yaml.safe_load(f)

    k_values = list(results.get("recall_at_k", {}).keys()) or list(
        results.get("config_snapshot", {}).get("evaluation", {}).get("k_values", [5, 10, 20])
    )
    k_values = [int(k) for k in k_values]

    df = pd.read_csv(pred_csv)

    ranked_items_per_user: dict = {}
    relevant_items_per_user: dict = {}
    for user_id, group in df.groupby("user_id"):
        ranked_items_per_user[user_id] = group.sort_values("y_pred", ascending=False)["business_id"].tolist()
        relevant_items_per_user[user_id] = set(group[group["y_true"] >= RELEVANCE_THRESHOLD]["business_id"])

    precision_k, recall_k, ndcg_k = precision_recall_ndcg_at_k(
        ranked_items_per_user, relevant_items_per_user, k_values
    )

    old_recall = results.get("recall_at_k", {})
    old_ndcg = results.get("ndcg_at_k", {})

    results["precision_at_k"] = {int(k): v for k, v in precision_k.items()}
    results["recall_at_k"] = {int(k): v for k, v in recall_k.items()}
    results["ndcg_at_k"] = {int(k): v for k, v in ndcg_k.items()}

    with open(yaml_path, "w") as f:
        yaml.safe_dump(results, f, allow_unicode=True)

    logger.info(
        "%s: recall@K %s -> %s | ndcg@K %s -> %s",
        yaml_path.name, old_recall, results["recall_at_k"], old_ndcg, results["ndcg_at_k"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute recall/NDCG@K dari predictions_*.csv (bugfix dedup)")
    parser.add_argument("--results-dir", type=str, default="checkpoints/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pred_files = sorted(results_dir.glob("predictions_*.csv"))
    logger.info("Ditemukan %d file predictions_*.csv di %s", len(pred_files), results_dir)

    for pred_csv in pred_files:
        recompute_one(pred_csv, results_dir)

    logger.info("Selesai. Jalankan ulang run_significance_test.py / build_manuscript_table.py "
                "utk regenerasi tabel ringkasan dgn angka recall/NDCG yang sudah benar.")


if __name__ == "__main__":
    main()
