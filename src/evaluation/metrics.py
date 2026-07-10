"""
src/evaluation/metrics.py

Modul evaluasi terpusat -- dipakai identik oleh SEMUA model yang dibandingkan
(baseline reimplementasi, SVD, Item-KNN, NCF, DeepFM, A2-FusionRS, dan
seluruh varian ablasi A1-A5). Sentralisasi ini penting agar tidak ada
perbedaan implementasi metrik antar model yang jadi confounding factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Hasil satu run (satu seed, satu domain) sebuah model, untuk agregasi
    multi-seed. `domain` WAJIB diisi kalau model yang sama dijalankan pada
    >1 domain (restoran & hotel) -- tanpa ini, aggregate_multi_seed_results()
    akan salah menggabungkan hasil restoran+hotel jadi satu baris."""

    model_name: str
    seed: int
    rmse: float
    mae: float
    domain: str = ""
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    per_sample_squared_errors: np.ndarray | None = None  # untuk uji signifikansi


def compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def precision_recall_ndcg_at_k(
    ranked_items_per_user: dict[int, list[int]],
    relevant_items_per_user: dict[int, set[int]],
    k_values: list[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Hitung Precision@K, Recall@K, NDCG@K rata-rata across user.

    Parameters
    ----------
    ranked_items_per_user : dict user_id -> list item_id terurut berdasarkan
        skor prediksi tertinggi (hasil ranking model, bukan urutan asli data).
    relevant_items_per_user : dict user_id -> set item_id yang dianggap
        "relevant" di ground truth (biasanya item dengan rating >= threshold,
        misal >=4, pada test set).
    """
    precision_results: dict[int, list[float]] = {k: [] for k in k_values}
    recall_results: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_results: dict[int, list[float]] = {k: [] for k in k_values}

    for user_id, ranked_items in ranked_items_per_user.items():
        relevant = relevant_items_per_user.get(user_id, set())
        if not relevant:
            continue  # user tanpa item relevant di test set diabaikan (standar praktik)

        for k in k_values:
            top_k = ranked_items[:k]
            hits = [1 if item in relevant else 0 for item in top_k]

            precision = sum(hits) / k
            recall = sum(hits) / len(relevant)

            dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
            ideal_hits = [1] * min(len(relevant), k)
            idcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(ideal_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precision_results[k].append(precision)
            recall_results[k].append(recall)
            ndcg_results[k].append(ndcg)

    avg_precision = {k: float(np.mean(v)) if v else 0.0 for k, v in precision_results.items()}
    avg_recall = {k: float(np.mean(v)) if v else 0.0 for k, v in recall_results.items()}
    avg_ndcg = {k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_results.items()}

    return avg_precision, avg_recall, avg_ndcg


def aggregate_multi_seed_results(results: list[RunResult]) -> pd.DataFrame:
    """Agregasi hasil multi-seed jadi mean +/- std per (model, domain), sesuai
    desain ablasi (>=3 seed per skenario). Group by (model_name, domain) --
    BUKAN model_name saja -- supaya hasil restoran & hotel dari model yang
    sama tidak tercampur jadi satu baris."""
    rows = []
    for r in results:
        row = {
            "model_name": r.model_name,
            "domain": r.domain,
            "seed": r.seed,
            "rmse": r.rmse,
            "mae": r.mae,
        }
        for k, v in r.precision_at_k.items():
            row[f"precision@{k}"] = v
        for k, v in r.recall_at_k.items():
            row[f"recall@{k}"] = v
        for k, v in r.ndcg_at_k.items():
            row[f"ndcg@{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("model_name", "domain", "seed")]
    summary = df.groupby(["model_name", "domain"])[metric_cols].agg(["mean", "std"])
    return summary


def save_predictions(path: str | Path, test_df: pd.DataFrame, y_pred: np.ndarray) -> None:
    """Simpan prediksi PER-SAMPEL (review_id, y_true, y_pred, squared_error)
    ke CSV -- data mentah yang dibutuhkan `significance_test()` (Wilcoxon)
    untuk membandingkan dua model secara berpasangan pada test set yang
    SAMA persis. Tanpa ini, metrik agregat (RMSE/MAE) saja TIDAK CUKUP untuk
    uji signifikansi -- harus run ulang semua skenario kalau baru dibutuhkan
    belakangan. Dipanggil identik oleh run_baseline.py, run_baseline_absa.py,
    dan run_classical_cf.py supaya formatnya konsisten lintas skenario.
    """
    out = pd.DataFrame(
        {
            "review_id": test_df["review_id"].values,
            "user_id": test_df["user_id"].values,
            "business_id": test_df["business_id"].values,
            "y_true": test_df["stars"].values,
            "y_pred": y_pred,
        }
    )
    out["squared_error"] = (out["y_true"] - out["y_pred"]) ** 2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("Prediksi per-sampel disimpan ke %s (untuk uji signifikansi nanti)", path)


def significance_test(
    errors_a: np.ndarray, errors_b: np.ndarray, test: str = "wilcoxon"
) -> tuple[float, float]:
    """Uji signifikansi perbedaan error antara dua model (misal A0 vs A1
    pada ablation study), dilakukan pada squared error per-sampel.

    Returns
    -------
    (statistic, p_value)
    """
    if len(errors_a) != len(errors_b):
        raise ValueError(
            "errors_a dan errors_b harus berpasangan (sampel identik) -- "
            "pastikan kedua model dievaluasi pada test set yang sama persis."
        )

    if test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(errors_a, errors_b)
    elif test == "paired_t":
        statistic, p_value = stats.ttest_rel(errors_a, errors_b)
    else:
        raise ValueError(f"test '{test}' tidak dikenal, gunakan 'wilcoxon' atau 'paired_t'")

    return float(statistic), float(p_value)


def sanity_check_rmse(rmse: float, rating_scale: tuple[float, float] = (1.0, 5.0)) -> None:
    """Peringatan otomatis jika RMSE mencurigakan rendah/tinggi.

    Ditambahkan khusus mengingat temuan RMSE 0.01-0.02 pada baseline paper
    yang diduga anomali metodologis (lih. diskusi sebelumnya) -- fungsi ini
    membantu mendeteksi dini kalau pipeline eksperimen kita sendiri
    mengalami masalah serupa (leakage, evaluasi pada train set, dsb).
    """
    scale_range = rating_scale[1] - rating_scale[0]
    if rmse < 0.05 * scale_range:
        logger.warning(
            "RMSE=%.4f SANGAT rendah relatif skala rating (%s). "
            "Ini mencurigakan -- periksa kemungkinan data leakage, evaluasi "
            "pada train set yang salah, atau target yang bocor ke fitur input "
            "sebelum melaporkan angka ini di manuskrip.",
            rmse,
            rating_scale,
        )
    elif rmse > 0.5 * scale_range:
        logger.warning(
            "RMSE=%.4f cukup tinggi relatif skala rating (%s). "
            "Periksa konvergensi training atau kemungkinan bug pada "
            "normalisasi/denormalisasi rating.",
            rmse,
            rating_scale,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Contoh cepat sanity check
    sanity_check_rmse(0.015)  # akan memicu warning, mensimulasikan kasus baseline
    sanity_check_rmse(0.85)   # RMSE realistis, tidak memicu warning
