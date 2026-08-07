"""
src/a2fusionrs/selection_split.py

Split INTERNAL train -> (train_fit, selection_dev) untuk SELEKSI kandidat
(hyperparameter/arsitektur) TANPA menyentuh test set.

MOTIVASI (Fix Temuan A2, reports/methodology_audit_2026-07-26.md): kalau
val_df dipakai GANDA -- early-stopping DAN pemilihan kandidat -- muncul bias
optimistik yang bergantung kandidat (kandidat yang "beruntung" di val
terpilih, lalu selisihnya tidak bertahan di test). Lebih parah lagi bila
kandidat dipilih langsung dari metrik TEST (itu p-hacking telanjang).

Pembagian peran yang benar dan dipakai proyek ini:
- `train_fit`      -> fitting model
- `val` (split asli) -> early-stopping SAJA
- `selection_dev`  -> membandingkan/memilih antar kandidat
- `test` (split asli) -> disentuh SEKALI di akhir, setelah kandidat dikunci

Fungsi ini semula hidup di `scripts/tune_deepmf_oof_val.py` (dipakai untuk
tuning DeepMF Fase 1). Dipindah ke `src/` supaya bisa dipakai ulang oleh
`run_attention_gated_fusion.py` (seleksi arsitektur A2-FusionRS Fase 2)
tanpa mengimpor script level-atas. Perilakunya TIDAK diubah sama sekali --
tes regresi `tests/test_split_train_fit_dev.py` tetap berlaku.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Porsi train_df asli yang disisihkan jadi selection_dev. Nilai default
# proyek (dipakai tuning DeepMF Fase 1 maupun seleksi arsitektur Fase 2)
# supaya kedua konteks konsisten.
SELECTION_DEV_FRACTION = 0.15


def split_train_fit_dev(
    train_df: pd.DataFrame, seed: int, dev_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pisah train_df asli jadi train_fit (fitting) + selection_dev (seleksi
    kandidat) -- SEKALI, deterministik thd seed, row-based (bukan user-based
    spt split_generator.py; ini split INTERNAL, TIDAK menyentuh/menggantikan
    file split bersama). Lihat docstring modul utk alasan lengkap (Fix
    Temuan A2).
    """
    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(len(train_df))
    n_dev = int(len(train_df) * dev_fraction)
    dev_idx = shuffled_idx[:n_dev]
    fit_idx = shuffled_idx[n_dev:]
    train_fit = train_df.iloc[fit_idx].reset_index(drop=True)
    selection_dev = train_df.iloc[dev_idx].reset_index(drop=True)
    logger.info(
        "split_train_fit_dev: train_fit=%d baris, selection_dev=%d baris (%.1f%% dari train asli %d baris)",
        len(train_fit), len(selection_dev), 100.0 * dev_fraction, len(train_df),
    )
    return train_fit, selection_dev
