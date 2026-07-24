"""
run_pyabsa_scoring.py

Stage 0 (A2-FusionRS Fase 2): skor ABSA berbasis model (PyABSA checkpoint
"english") untuk SELURUH review di satu domain, SEKALI (bukan per seed/
skenario ablasi) -- lihat alasan di src/a2fusionrs/pyabsa_scorer.py dan
estimasi biaya di phase2_notes/attention_gated_fusion_design.md Bagian 4.

BERBEDA dari run_baseline.py/run_baseline_absa.py: script ini TIDAK melatih
model apa pun (DeepMF/CBF/fusion) dan TIDAK mengevaluasi RMSE -- murni
menghasilkan cache skor ABSA yang akan dikonsumsi run_attention_gated_fusion.py
(Stage 3) utk semua seed & skenario secara berulang.

PRASYARAT: split domain (train/val/test) SUDAH ADA (split_generator.py) --
script ini load-only, sama seperti run_classical_cf.py/run_baseline_absa.py.

Usage:
    # Verifikasi murah dulu (WAJIB sebelum run penuh) -- cocokkan dgn
    # benchmark yg sudah terdokumentasi di pyabsa_investigation.md
    # (70,0% cakupan, 2,77 rata-rata aspek/review, checkpoint "english"):
    python run_pyabsa_scoring.py --config configs/tripadvisor_hotel_config_colab.yaml \
        --sample-size 500 --random-state 42

    # Run penuh 1 domain (SETELAH verifikasi sample di atas cocok):
    python run_pyabsa_scoring.py --config configs/tripadvisor_hotel_config_colab.yaml
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd

from src.legacy.a2fusionrs.pyabsa_scorer import PyABSAAspectScorer, PyABSAConfig
from src.config_utils import load_config
from src.preprocessing import TextPreprocessor
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _update_coverage_summary(
    config: dict, domain: str, coverage: dict, checkpoint: str
) -> None:
    """Simpan/perbarui 1 baris ringkasan cakupan aspek PyABSA per domain ke
    file bersama -- SEBELUM ini, angka cakupan cuma muncul di log Colab yang
    sifatnya sementara (hilang begitu sesi ditutup), padahal angka ini
    kemungkinan besar akan dikutip di manuskrip A2-FusionRS (setara Table II
    di paper A2-IRM, tapi utk cakupan model-based bukan keyword-matching).

    Path SAMA dengan direktori `results/` bersama yang sudah dipakai
    run_baseline.py/run_baseline_absa.py (Path(checkpoint_dir).parent /
    "results") -- supaya semua ringkasan hasil ada di 1 tempat yang sama,
    bukan folder baru terpisah.

    HANYA dipanggil utk run PENUH (bukan mode --sample-size verifikasi) --
    baris existing utk domain yang sama DITIMPA (bukan ditambah duplikat)
    supaya re-run domain yang sama tidak menghasilkan baris ganda.
    """
    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "pyabsa_coverage_summary.csv"

    new_row = {
        "domain": domain,
        "checkpoint": checkpoint,
        "n_reviews": coverage["n_reviews"],
        "n_with_any_aspect": coverage["n_with_any_aspect"],
        "pct_with_any_aspect": round(coverage["pct_with_any_aspect"] * 100, 2),
        "avg_aspects_per_review": round(coverage["avg_aspects_per_review"], 2),
        "computed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["domain"] != domain]
        updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        updated = pd.DataFrame([new_row])

    updated = updated.sort_values("domain").reset_index(drop=True)
    updated.to_csv(summary_path, index=False)
    logger.info("Ringkasan cakupan PyABSA (%s) diperbarui di %s.", domain, summary_path)


def run_scoring(config: dict, sample_size: int | None, random_state: int, checkpoint: str) -> None:
    exp_cfg = config["experiment"]
    split_cfg = config["split"]
    domain = exp_cfg["domain"]

    # ---------- 1. Load split (WAJIB sudah ada) & gabungkan SEMUA baris --
    # skor PyABSA tidak bergantung train/val/test -- kita butuh SELURUH
    # review domain ini, bukan satu split saja.
    logger.info("=== Memuat split domain '%s' (WAJIB sudah ada) ===", domain)
    split_output_dir = Path(split_cfg["output_dir"])
    splits = UserBasedSplitGenerator.load(split_output_dir)
    full_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    logger.info("Total review domain '%s': %d baris (train+val+test digabung).", domain, len(full_df))

    # ---------- 2. Preprocessing (reuse, sama persis dgn pipeline lain) ----------
    logger.info("=== Preprocessing teks ===")
    preprocessor = TextPreprocessor()
    full_df = preprocessor.preprocess_dataframe(full_df)

    if sample_size is not None:
        full_df = full_df.sample(n=min(sample_size, len(full_df)), random_state=random_state).reset_index(
            drop=True
        )
        logger.info(
            "Mode VERIFIKASI: subsample %d review (random_state=%d) -- BUKAN run penuh. "
            "Bandingkan coverage_report() di bawah dgn benchmark pyabsa_investigation.md "
            "SEBELUM menjalankan run penuh domain ini.",
            len(full_df),
            random_state,
        )

    # ---------- 3. Skoring PyABSA ----------
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / "pyabsa"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_sample{sample_size}_seed{random_state}" if sample_size is not None else ""
    cache_path = checkpoint_dir / f"pyabsa_scores_{domain}{suffix}.csv"

    if cache_path.exists():
        logger.info(
            "Cache skor PyABSA SUDAH ADA di %s -- tidak menjalankan inferensi ulang. "
            "Hapus file ini manual kalau memang ingin re-skor (mis. checkpoint diganti).",
            cache_path,
        )
        return

    scorer = PyABSAAspectScorer(PyABSAConfig(checkpoint=checkpoint))
    scored_df = scorer.score_dataframe(full_df, text_column="text_bert", review_id_column="review_id")

    coverage = scorer.coverage_report(scored_df)
    logger.info("=== Cakupan aspek PyABSA (checkpoint '%s', domain '%s') ===", checkpoint, domain)
    logger.info(
        "%d/%d review (%.1f%%) punya >=1 aspek, rata-rata %.2f aspek/review.",
        coverage["n_with_any_aspect"],
        coverage["n_reviews"],
        coverage["pct_with_any_aspect"] * 100,
        coverage["avg_aspects_per_review"],
    )

    scored_df.to_csv(cache_path, index=False)
    logger.info("Skor PyABSA disimpan ke cache %s.", cache_path)

    if sample_size is None:
        # Ringkasan cakupan HANYA disimpan utk run penuh -- angka dari mode
        # verifikasi --sample-size bukan angka kanonis domain ini, cuma
        # utk sanity check sebelum commit ke run penuh.
        _update_coverage_summary(config, domain, coverage, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skor ABSA berbasis model (PyABSA) untuk seluruh review 1 domain (Stage 0 Fase 2)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Kalau diberikan, skor cuma subsample (mode verifikasi murah -- cocokkan "
        "coverage_report() dgn benchmark pyabsa_investigation.md SEBELUM run penuh). "
        "Kalau tidak diberikan, skor SELURUH review domain (run penuh, ~jam-an di GPU).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="english",
        help="Checkpoint PyABSA ('english' direkomendasikan utk domain proyek ini -- "
        "lihat pyabsa_investigation.md).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_scoring(cfg, args.sample_size, args.random_state, args.checkpoint)
