"""
run_attention_gated_fusion.py

A2-FusionRS Fase 2 (Stage 3 rencana implementasi): pipeline Attention-Gated
Fusion, DUPLIKAT run_baseline_absa.py (load -> split -> preprocess -> DeepMF
-> CBF -> fusion -> evaluasi), TAPI:
- Tahap 5 (DeepMF) & 6 (CBF) memakai method BARU predict_with_latent()/
  predict_features() (Stage 1) -- mengekspos representasi VEKTOR, bukan
  cuma skalar prediksi akhir seperti run_baseline_absa.py.
- Tahap 4 (sentimen) mendukung 2 sumber ABSA: 'keyword' (reuse
  KeywordAspectSentimentScorer, sama seperti run_baseline_absa.py mode
  concat_confidence) atau 'pyabsa' (load cache Stage 0 + vectorize_absa_
  features(), lihat src/a2fusionrs/pyabsa_scorer.py).
- Tahap 7 (fusion) DIGANTI: bukan NMF+DecisionTree, tapi salah satu dari
  beberapa skenario (lihat AGF_SCENARIOS di bawah) -- dipilih via
  `--scenario`, BUKAN axis config YAML terpisah (lihat alasan di
  attention_gated_fusion_design.md & catatan "Hindari ledakan file config"
  di plan implementasi).

Tahap 1-3 (load split, preprocess) disalin verbatim dari run_baseline_absa.py
-- sengaja TIDAK di-refactor jadi shared/pluggable, demi meminimalkan risiko
ke pipeline yang sudah tervalidasi (prinsip yang sama dipegang
run_baseline_absa.py sendiri terhadap run_baseline.py).

PRASYARAT:
1. Split domain WAJIB sudah ada (split_generator.py).
2. Checkpoint model SA (`sentiment_bert/`) WAJIB sudah ada dari
   run_baseline.py (dipakai scorer keyword ABSA, DAN sbg fallback skor
   whole-review untuk baris yang PyABSA gagal temukan aspek).
3. Untuk --absa-source pyabsa: cache skor PyABSA WAJIB sudah ada dari
   run_pyabsa_scoring.py (Stage 0) -- script ini TIDAK menjalankan
   inferensi PyABSA baru sama sekali (mahal, ~jam-an di GPU).

Usage:
    python run_attention_gated_fusion.py \
        --config configs/tripadvisor_hotel_config_agf_colab.yaml \
        --scenario full_agf --seed 42
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.a2fusionrs.absa_bert import ABSAConfig, KeywordAspectSentimentScorer
from src.a2fusionrs.attention_gated_fusion import AGFConfig, AttentionGatedFusionTrainer
from src.a2fusionrs.pyabsa_scorer import (
    ABSA_VECTOR_FEATURE_NAMES,
    build_aspect_sequences,
    build_aspect_vocab,
    load_cached_scores,
    vectorize_absa_features,
    vectorize_absa_features_rich,
)
from src.baseline.cbf_clustering import CBFConfig, CBFPredictor
from src.baseline.deepmf import DeepMFConfig, DeepMFTrainer, InteractionDataset
from src.baseline.fusion_nmf_dt import FusionConfig, NMFDecisionTreeFusion
from src.baseline.sentiment_bert import GlobalSentimentBERT, SentimentBertConfig
from src.config_utils import load_config
from src.evaluation.metrics import (
    compute_rmse_mae,
    precision_recall_ndcg_at_k,
    sanity_check_rmse,
    save_predictions,
    save_results_yaml,
)
from src.preprocessing import TextPreprocessor
from src.split_generator import UserBasedSplitGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dispatch skenario Tier 1+2 -- SATU class AttentionGatedFusionModel (Stage 2)
# melayani semua varian AGF via kombinasi modalitas aktif + 2 flag, BUKAN
# duplikasi kelas per skenario (lihat docstring attention_gated_fusion.py).
# "static_pyabsa" dan "weighted_avg" DILUAR dict ini -- ditangani terpisah
# di run_pipeline() karena tidak memakai AttentionGatedFusionTrainer sama
# sekali (lihat Tahap 7 di bawah).
AGF_SCENARIOS: dict[str, dict] = {
    "full_agf": dict(modalities=["deepmf", "cbf", "absa"], use_attention=True, pooling="gate"),
    "attention_only": dict(modalities=["deepmf", "cbf", "absa"], use_attention=True, pooling="mean"),
    "gating_only": dict(modalities=["deepmf", "cbf", "absa"], use_attention=False, pooling="gate"),
    "agf_keyword": dict(
        modalities=["deepmf", "cbf", "absa"], use_attention=True, pooling="gate", absa_source_override="keyword"
    ),
    "leave_out_deepmf": dict(modalities=["cbf", "absa"], use_attention=True, pooling="gate"),
    "leave_out_cbf": dict(modalities=["deepmf", "absa"], use_attention=True, pooling="gate"),
    "leave_out_absa": dict(modalities=["deepmf", "cbf"], use_attention=True, pooling="gate"),
    "concat_mlp": dict(modalities=["deepmf", "cbf", "absa"], use_attention=False, pooling="concat"),
}
# static_keyword_pyabsa: KONTROL ATRIBUSI (Stage 7+) -- tree NMF+DT (sama
# spt A2-IRM) TAPI diberi keyword ABSA + PyABSA-rich sekaligus. Menjawab
# "apakah tree juga membaik dgn PyABSA, atau cuma AGF?" -- kalau tree+pyabsa
# ~= AGF+pyabsa, PyABSA-nya yg komplementer (AGF bukan mekanisme unik);
# kalau tree+pyabsa jauh lebih buruk, AGF mengeksploitasi PyABSA lebih baik.
NON_AGF_SCENARIOS = ("static_pyabsa", "weighted_avg", "static_keyword_pyabsa")
ALL_SCENARIOS = list(AGF_SCENARIOS) + list(NON_AGF_SCENARIOS)


def _compute_absa_features(
    config: dict,
    exp_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sa_checkpoint_dir: Path,
    absa_source: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Kembalikan dict {'train'/'val'/'test': array vektor ABSA} + dict
    tambahan berisi `sentiment_score` skalar per split (dipakai CBF's
    sentiment_agg & skenario 'weighted_avg') -- dipisah dari alur utama
    run_pipeline() karena tahap ini punya percabangan sumber ABSA yang
    cukup panjang (keyword vs pyabsa)."""
    sa_config = SentimentBertConfig(
        model_name=config["sentiment_baseline"]["model_name"],
        max_length=config["data"].get("max_seq_length", 512),
        batch_size=config["sentiment_baseline"]["batch_size"],
        learning_rate=config["sentiment_baseline"]["learning_rate"],
        epochs=config["sentiment_baseline"]["epochs"],
    )
    if not (sa_checkpoint_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Checkpoint model SA tidak ditemukan di {sa_checkpoint_dir}. "
            "run_attention_gated_fusion.py TIDAK melakukan training BERT baru -- "
            "jalankan run_baseline.py (config domain yang sama, logging."
            "checkpoint_dir yang sama) sampai tahap 4 selesai terlebih dahulu."
        )
    logger.info("Memuat model SA dari checkpoint %s (dipakai ulang, tanpa training baru).", sa_checkpoint_dir)
    sa_model = GlobalSentimentBERT.load(str(sa_checkpoint_dir), sa_config)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    if absa_source == "keyword":
        # Reuse PERSIS varian terbaik Fase 1 (Concat+Confidence) sbg
        # representasi ABSA keyword -- sel faktorial "AGF + Keyword".
        absa_config = ABSAConfig(domain=exp_cfg["domain"])
        scorer = KeywordAspectSentimentScorer(sa_model, absa_config)
        aspect_names = list(scorer.aspect_keywords.keys())
        confidence_names = [f"{a}_confidence" for a in aspect_names]

        absa_cache = sa_checkpoint_dir / "absa_concat_confidence_scores.csv"
        if absa_cache.exists():
            logger.info("Cache skor ABSA-concat-confidence ditemukan di %s -- skip inference.", absa_cache)
            cache_df = pd.read_csv(absa_cache).set_index("review_id")
            for part_df in splits.values():
                for col in aspect_names + confidence_names:
                    part_df[col] = part_df["review_id"].map(cache_df[col])
        else:
            for name, part_df in splits.items():
                logger.info("=== ABSA-keyword: menghitung skor per-aspek split '%s' ===", name)
                t0 = time.time()
                aspect_df = part_df.pipe(scorer.score_dataframe_per_aspect)
                for aspect in aspect_names:
                    part_df[aspect] = aspect_df[aspect].values
                evidence_df = scorer.compute_aspect_evidence_counts(part_df)
                for aspect, conf_col in zip(aspect_names, confidence_names):
                    sentiment_conf = part_df[aspect].apply(scorer._sentiment_confidence)
                    evidence_conf = evidence_df[aspect].apply(scorer._evidence_confidence)
                    part_df[conf_col] = ((sentiment_conf + evidence_conf) / 2.0).values
                logger.info(
                    "=== ABSA-keyword: split '%s' selesai dalam %.1f menit ===", name, (time.time() - t0) / 60
                )
            pd.concat([p[["review_id"] + aspect_names + confidence_names] for p in splits.values()]).to_csv(
                absa_cache, index=False
            )
            logger.info("Skor ABSA-keyword disimpan ke cache %s.", absa_cache)

        feature_cols = aspect_names + confidence_names
        absa_features = {name: p[feature_cols].values.astype(np.float32) for name, p in splits.items()}
        for part_df in splits.values():
            part_df["sentiment_score"] = part_df[aspect_names].mean(axis=1)
        return {"features": absa_features, "splits": splits}

    if absa_source == "pyabsa":
        pyabsa_cache_dir = Path(config["logging"]["checkpoint_dir"]) / "pyabsa"
        cache_path = pyabsa_cache_dir / f"pyabsa_scores_{exp_cfg['domain']}.csv"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache skor PyABSA tidak ditemukan di {cache_path}. Jalankan "
                "run_pyabsa_scoring.py (Stage 0) untuk domain ini TERLEBIH DAHULU -- "
                "script ini TIDAK menjalankan inferensi PyABSA baru (mahal, ~jam-an di GPU)."
            )
        pyabsa_df = load_cached_scores(str(cache_path)).set_index("review_id")

        # Cache TERPISAH utk skor fallback SA-BERT whole-review (baris yang
        # PyABSA gagal temukan aspek) -- keyed by review_id, SEKALI dihitung
        # per domain, dipakai ulang lintas SEMUA seed & skenario yang
        # memakai absa_source='pyabsa'. TANPA cache ini, ~150 kombinasi
        # skenario x domain x seed di Stage 6 akan mengulang inferensi
        # SA-BERT yang sama berkali-kali (mahal & sia-sia -- fallback score
        # 1 review TIDAK bergantung skenario/seed, sama seperti skor PyABSA
        # sendiri).
        fallback_cache_path = pyabsa_cache_dir / f"sa_fallback_scores_{exp_cfg['domain']}.csv"
        if fallback_cache_path.exists():
            fallback_cache = pd.read_csv(fallback_cache_path).set_index("review_id")["fallback_score"].to_dict()
        else:
            fallback_cache = {}

        absa_features = {}
        for name, part_df in splits.items():
            merged = part_df[["review_id", "text_bert"]].merge(
                pyabsa_df[["n_aspects", "aspects", "sentiments", "confidences", "probs"]],
                left_on="review_id",
                right_index=True,
                how="left",
            )
            # Baris yang review_id-nya TIDAK ada di cache PyABSA (mis. split
            # dibuat setelah scoring, atau subsample verifikasi Stage 0) --
            # diperlakukan sbg 0 aspek (bukan crash), fallback whole-review
            # tetap akan mengisi via sa_model di bawah.
            missing = merged["n_aspects"].isna()
            if missing.any():
                logger.warning(
                    "%d/%d baris split '%s' TIDAK ada di cache PyABSA -- diisi 0 aspek "
                    "(fallback whole-review SA-BERT). Pastikan run_pyabsa_scoring.py "
                    "dijalankan TANPA --sample-size (run penuh) sebelum run skala penuh.",
                    int(missing.sum()),
                    len(merged),
                    name,
                )
                merged.loc[missing, "n_aspects"] = 0
                for col in ("aspects", "sentiments", "confidences", "probs"):
                    merged.loc[missing, col] = merged.loc[missing, col].apply(
                        lambda x: [] if not isinstance(x, list) else x
                    )
            merged["n_aspects"] = merged["n_aspects"].astype(int)

            # Fallback whole-review HANYA utk baris 0-aspek (bukan semua
            # baris) -- jauh lebih murah dari re-run ABSA penuh, dan
            # menjaga paralel metodologis dgn fallback keyword ABSA.
            zero_aspect_rows = merged[merged["n_aspects"] == 0]
            if len(zero_aspect_rows) > 0:
                uncached_rows = zero_aspect_rows[~zero_aspect_rows["review_id"].isin(fallback_cache)]
                logger.info(
                    "Split '%s': %d/%d baris (0 aspek PyABSA) butuh fallback SA-BERT whole-review "
                    "(%d sudah ada di cache, %d baru dihitung).",
                    name,
                    len(zero_aspect_rows),
                    len(merged),
                    len(zero_aspect_rows) - len(uncached_rows),
                    len(uncached_rows),
                )
                if len(uncached_rows) > 0:
                    new_preds = sa_model.predict_proba(uncached_rows["text_bert"].fillna("").tolist())
                    fallback_cache.update(dict(zip(uncached_rows["review_id"], (float(p) for p in new_preds))))

            fallback_scores = {rid: fallback_cache[rid] for rid in zero_aspect_rows["review_id"] if rid in fallback_cache}
            vec = vectorize_absa_features(merged, fallback_scores=fallback_scores)
            absa_features[name] = vec.astype(np.float32)
            # sentiment_score turunan utk CBF's sentiment_agg -- pakai kolom
            # ringkasan mean_positive_prob (index 1 di ABSA_VECTOR_FEATURE_NAMES).
            part_df["sentiment_score"] = vec[:, ABSA_VECTOR_FEATURE_NAMES.index("mean_positive_prob")]

        pd.DataFrame(
            {"review_id": list(fallback_cache.keys()), "fallback_score": list(fallback_cache.values())}
        ).to_csv(fallback_cache_path, index=False)
        logger.info("Cache fallback SA-BERT (%d review_id) disimpan ke %s.", len(fallback_cache), fallback_cache_path)

        return {"features": absa_features, "splits": splits}

    raise ValueError(f"absa_source '{absa_source}' tidak dikenal -- pakai 'keyword' atau 'pyabsa'.")


def _compute_pyabsa_rich_modality(
    config: dict, exp_cfg: dict, splits: dict, rich: bool = True
) -> dict[str, np.ndarray]:
    """Fitur PyABSA per-aspek (rich order-statistics kalau rich=True, atau
    5-dim ringkasan kalau False) sbg MODALITAS EKSTRA utk AGF -- dipakai di
    atas keyword-ABSA (agf_keyword) supaya AGF punya sinyal per-aspek BARU
    yg tree A2-IRM tak punya. Reuse cache PyABSA + cache fallback SA-BERT yg
    SUDAH ADA (Stage 0 & Stage 6) -- TIDAK memuat model apa pun (cepat).
    """
    pyabsa_cache_dir = Path(config["logging"]["checkpoint_dir"]) / "pyabsa"
    cache_path = pyabsa_cache_dir / f"pyabsa_scores_{exp_cfg['domain']}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache skor PyABSA tidak ditemukan di {cache_path} -- jalankan run_pyabsa_scoring.py dulu."
        )
    pyabsa_df = load_cached_scores(str(cache_path)).set_index("review_id")

    fb_path = pyabsa_cache_dir / f"sa_fallback_scores_{exp_cfg['domain']}.csv"
    fb_cache = (
        pd.read_csv(fb_path).set_index("review_id")["fallback_score"].to_dict() if fb_path.exists() else {}
    )

    vectorizer = vectorize_absa_features_rich if rich else vectorize_absa_features
    out = {}
    for name, part_df in splits.items():
        merged = part_df[["review_id"]].merge(
            pyabsa_df[["n_aspects", "aspects", "sentiments", "confidences", "probs"]],
            left_on="review_id", right_index=True, how="left",
        )
        missing = merged["n_aspects"].isna()
        if missing.any():
            merged.loc[missing, "n_aspects"] = 0
            for col in ("aspects", "sentiments", "confidences", "probs"):
                merged[col] = merged[col].apply(lambda x: x if isinstance(x, list) else [])
        merged["n_aspects"] = merged["n_aspects"].astype(int)
        out[name] = vectorizer(merged, fallback_scores=fb_cache).astype(np.float32)
    logger.info(
        "Modalitas EKSTRA PyABSA (%s, %d fitur) dihitung utk semua split.",
        "rich" if rich else "5-dim", out["train"].shape[1],
    )
    return out


def _compute_pyabsa_aspect_sequences(
    config: dict, exp_cfg: dict, splits: dict, max_aspects: int = 8, vocab_top_k: int = 500
) -> tuple[dict, dict]:
    """Jalur X: sequence aspek PyABSA panjang-variabel + vocab identitas aspek,
    utk AspectSequencePooling di AGF. Vocab dibangun HANYA dari TRAIN (cegah
    leakage istilah aspek dari test). Return (vocab, {split: {ids,feats,mask}}).
    """
    pyabsa_cache_dir = Path(config["logging"]["checkpoint_dir"]) / "pyabsa"
    cache_path = pyabsa_cache_dir / f"pyabsa_scores_{exp_cfg['domain']}.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache skor PyABSA tidak ditemukan di {cache_path} -- jalankan run_pyabsa_scoring.py dulu."
        )
    pyabsa_df = load_cached_scores(str(cache_path)).set_index("review_id")
    fb_path = pyabsa_cache_dir / f"sa_fallback_scores_{exp_cfg['domain']}.csv"
    fb_cache = (
        pd.read_csv(fb_path).set_index("review_id")["fallback_score"].to_dict() if fb_path.exists() else {}
    )

    def merged_for(part_df):
        m = part_df[["review_id"]].merge(
            pyabsa_df[["n_aspects", "aspects", "sentiments", "confidences", "probs"]],
            left_on="review_id", right_index=True, how="left",
        )
        miss = m["n_aspects"].isna()
        if miss.any():
            m.loc[miss, "n_aspects"] = 0
            for col in ("aspects", "sentiments", "confidences", "probs"):
                m[col] = m[col].apply(lambda x: x if isinstance(x, list) else [])
        m["n_aspects"] = m["n_aspects"].astype(int)
        return m

    merged = {name: merged_for(part_df) for name, part_df in splits.items()}
    vocab = build_aspect_vocab(merged["train"], top_k=vocab_top_k)
    seqs = {}
    for name, m in merged.items():
        ids, feats, mask = build_aspect_sequences(m, vocab, max_aspects=max_aspects, fallback_scores=fb_cache)
        seqs[name] = {"ids": ids, "feats": feats, "mask": mask}
    logger.info(
        "Jalur X: sequence aspek PyABSA (vocab=%d istilah, max_aspects=%d) dihitung utk semua split.",
        len(vocab), max_aspects,
    )
    return vocab, seqs


def _export_interpretability(
    agf_trainer, test_features, base_test, aspect_seq_test, aspect_vocab,
    test_df, test_preds, rating_scale, out_dir, stem,
) -> None:
    """§6.5 interpretability: (B) studi kasus atensi per-aspek + (C) uji
    faithfulness perturbasi (buang aspek paling-diperhatikan vs aspek acak,
    ukur |Δprediksi|). Base statis tak bergantung aspek, jadi ablasi hanya
    mengubah KOREKSI AGF -- mengisolasi pengaruh aspek pd refinement (jujur:
    ini menjelaskan koreksi di atas base, bukan seluruh prediksi)."""
    ids = aspect_seq_test["ids"]                 # (N, L)
    feats = aspect_seq_test["feats"]             # (N, L, 4) = [P_neg,P_neu,P_pos,conf]
    mask = np.asarray(aspect_seq_test["mask"]).astype(bool)  # (N, L)
    attn = agf_trainer.extract_aspect_attention(test_features, base_test, aspect_seq_test)  # (N, L)
    id2term = {v: k for k, v in aspect_vocab.items()}  # id 0=PAD, 1=UNK
    N = ids.shape[0]
    n_valid = mask.sum(axis=1)

    # ----- (B) tabel studi kasus (semua baris; filter analisis di skrip lain) -----
    rows = []
    for i in range(N):
        valid = np.where(mask[i])[0]
        terms = [id2term.get(int(ids[i, j]), "<UNK>") for j in valid]
        rows.append({
            "review_id": test_df["review_id"].values[i],
            "n_aspects": int(n_valid[i]),
            "pred": round(float(test_preds[i]), 4),
            "actual": float(test_df["stars"].values[i]),
            "aspects": "|".join(terms),
            "attn": "|".join(f"{a:.3f}" for a in attn[i, valid]),
            "p_pos": "|".join(f"{p:.2f}" for p in feats[i, valid, 2]),
            "p_neg": "|".join(f"{p:.2f}" for p in feats[i, valid, 0]),
        })
    case_path = out_dir / f"interp_cases_{stem}.csv"
    pd.DataFrame(rows).to_csv(case_path, index=False)
    logger.info("§6.5 Exp-B: studi kasus atensi aspek disimpan ke %s (%d baris).", case_path, N)

    # ----- (C) faithfulness: ablasi top-atensi vs acak (baris >=2 aspek) -----
    multi = np.where(n_valid >= 2)[0]
    if len(multi) == 0:
        logger.warning("§6.5 Exp-C dilewati: tak ada baris dgn >=2 aspek valid.")
        return
    attn_masked = np.where(mask, attn, -np.inf)
    top_idx = attn_masked.argmax(axis=1)
    rng = np.random.default_rng(0)
    mask_top, mask_rand = mask.copy(), mask.copy()
    for i in multi:
        mask_top[i, top_idx[i]] = False
        valid = np.where(mask[i])[0]
        r = rng.choice(valid[valid != top_idx[i]])
        mask_rand[i, r] = False

    def _pred(mask_variant):
        aseq = {"ids": ids, "feats": feats, "mask": mask_variant}
        p, _ = agf_trainer.predict(test_features, rating_scale, base_norm=base_test, aspect_seq=aseq)
        return np.clip(p, rating_scale[0], rating_scale[1])

    delta_top = np.abs(_pred(mask_top) - test_preds)[multi]
    delta_rand = np.abs(_pred(mask_rand) - test_preds)[multi]
    frac_bigger = float((delta_top > delta_rand).mean())
    from scipy.stats import wilcoxon
    try:
        _, p_val = wilcoxon(delta_top, delta_rand)
    except ValueError:
        p_val = float("nan")
    summary = {
        "n_rows_ge2_aspects": int(len(multi)),
        "mean_delta_top_attended": round(float(delta_top.mean()), 5),
        "mean_delta_random": round(float(delta_rand.mean()), 5),
        "frac_top_gt_random": round(frac_bigger, 4),
        "wilcoxon_p": p_val,
    }
    faith_path = out_dir / f"interp_faithfulness_{stem}.csv"
    pd.DataFrame([summary]).to_csv(faith_path, index=False)
    logger.info(
        "§6.5 Exp-C faithfulness: |Δ|top=%.5f vs acak=%.5f (top>acak di %.1f%% baris, "
        "Wilcoxon p=%.2e). Disimpan ke %s.",
        summary["mean_delta_top_attended"], summary["mean_delta_random"],
        frac_bigger * 100, p_val, faith_path,
    )


def run_pipeline(
    config: dict,
    scenario: str,
    input_standardize: bool = False,
    use_scalar_preds: bool = False,
    representation: str = "vector",
    residual_base: str = "none",
    extra_pyabsa: str = "none",
    run_tag: str = "",
    export_interpretability: bool = False,
) -> None:
    """`input_standardize` & `use_scalar_preds` adalah 2 DIAGNOSTIK Stage 7
    (default False = perilaku ASLI 150-run, byte-identical). Keduanya HANYA
    berlaku untuk skenario AGF (bukan static_pyabsa/weighted_avg):

    - input_standardize: StandardScaler per-modalitas (fit di train, apply ke
      semua split) SEBELUM masuk AGF. Alasan: laten DeepMF (pasca-ReLU),
      fitur CBF (PCA, bisa +-besar), dan ABSA ([0,1]) ada di SKALA sangat
      berbeda tanpa normalisasi -- diduga penyebab 1 modalitas "tenggelam"
      (leave_out_deepmf ~= full_agf) & anomali gate lintas domain.
    - use_scalar_preds: umpankan prediksi rating skalar DeepMF/CBF (yg SUDAH
      terkalibrasi, dipakai A2-IRM & static_pyabsa) sbg fitur tambahan di
      modalitas masing-masing -- AGF versi asli MEMBUANG ini, cuma pakai
      laten/fitur mentah.

    `representation` ("vector"|"asymmetric") & `residual_base` ("none"|
    "static_fusion") adalah REDESIGN Stage 7+ (hasil analisis akar Gap 2):
    - representation="asymmetric": DeepMF/CBF masuk sbg PREDIKSI SKALAR
      ternormalisasi (sinyal terkuat mereka -- mereka prediktor rating,
      laten mentahnya harus di-decode ulang oleh jaringan kecil), ABSA
      tetap vektor kaya (satu-satunya modalitas yg benar diuntungkan
      representasi kaya).
    - residual_base="static_fusion": base = prediksi NMF+DecisionTree (sama
      seperti A2-IRM) dihitung dari [absa, deepmf_scalar, cbf_scalar]; AGF
      belajar KOREKSI adaptif di atasnya. By-construction >= base, jadi
      target "kalahkan A2-IRM" tinggal soal apakah koreksi menambah nilai.

    `run_tag`: suffix ke nama file hasil supaya run diagnostik TIDAK menimpa
    hasil utama 150-run (mis. run_tag='norm' -> agf_full_agf_norm_...yaml).
    """
    if representation not in ("vector", "asymmetric"):
        raise ValueError(f"representation '{representation}' -- pakai 'vector' atau 'asymmetric'.")
    if residual_base not in ("none", "static_fusion", "static_fusion_oof"):
        raise ValueError(
            f"residual_base '{residual_base}' -- pakai 'none'/'static_fusion'/'static_fusion_oof'."
        )
    if extra_pyabsa not in ("none", "rich", "summary", "perseq"):
        raise ValueError(f"extra_pyabsa '{extra_pyabsa}' -- pakai 'none'/'rich'/'summary'/'perseq'.")
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"scenario '{scenario}' tidak dikenal -- pilih salah satu dari {ALL_SCENARIOS}.")

    exp_cfg = config["experiment"]
    split_cfg = config["split"]

    np.random.seed(exp_cfg["seed"])
    torch.manual_seed(exp_cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(exp_cfg["seed"])

    # ---------- 2. Split (WAJIB sudah ada, load-only) ----------
    logger.info("=== Tahap 2: Memuat split (WAJIB sudah ada) ===")
    split_output_dir = Path(split_cfg["output_dir"])
    splits_raw = UserBasedSplitGenerator.load(split_output_dir)
    train_df, val_df, test_df = splits_raw["train"], splits_raw["val"], splits_raw["test"]

    # ---------- 3. Preprocessing ----------
    logger.info("=== Tahap 3: Preprocessing teks ===")
    preprocessor = TextPreprocessor()
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)

    # ---------- 4. ABSA (sumber fitur modalitas 'absa') ----------
    logger.info("=== Tahap 4: Skor ABSA (sumber modalitas 'absa') ===")
    sa_checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / "sentiment_bert"
    default_absa_source = config.get("agf", {}).get("absa_source", "pyabsa")
    absa_source = AGF_SCENARIOS.get(scenario, {}).get("absa_source_override", default_absa_source)
    if scenario == "static_keyword_pyabsa":
        absa_source = "keyword"  # kontrol pakai keyword ABSA (spt A2-IRM) + PyABSA-rich di tree
    logger.info("Sumber ABSA untuk skenario '%s': '%s'", scenario, absa_source)

    absa_result = _compute_absa_features(
        config, exp_cfg, train_df, val_df, test_df, sa_checkpoint_dir, absa_source
    )
    absa_features = absa_result["features"]

    # ---------- 5. DeepMF (predict_with_latent, Stage 1) ----------
    logger.info("=== Tahap 5: Training DeepMF ===")
    all_users = pd.concat([train_df["user_id"], val_df["user_id"], test_df["user_id"]]).unique()
    all_items = pd.concat(
        [train_df["business_id"], val_df["business_id"], test_df["business_id"]]
    ).unique()
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {b: i for i, b in enumerate(all_items)}

    deepmf_config = DeepMFConfig(
        embedding_dim=config["deepmf"]["embedding_dim"],
        hidden_layers=tuple(config["deepmf"]["hidden_layers"]),
        dropout=config["deepmf"]["dropout"],
        batch_size=config["deepmf"]["batch_size"],
        learning_rate=config["deepmf"]["learning_rate"],
        negative_sampling_ratio=config["deepmf"]["negative_sampling_ratio"],
    )
    train_interactions = InteractionDataset(
        train_df, user2idx, item2idx, len(all_items), deepmf_config.negative_sampling_ratio, seed=exp_cfg["seed"]
    )
    val_interactions = InteractionDataset(
        val_df, user2idx, item2idx, len(all_items), negative_ratio=0, seed=exp_cfg["seed"]
    )
    deepmf_trainer = DeepMFTrainer(len(all_users), len(all_items), deepmf_config)
    deepmf_trainer.fit(train_interactions, val_interactions)

    rating_scale = (1.0, 5.0)
    deepmf_scalar = {}
    deepmf_latent = {}
    for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        scalar, latent = deepmf_trainer.predict_with_latent(part_df, user2idx, item2idx, rating_scale)
        deepmf_scalar[name] = scalar
        deepmf_latent[name] = latent

    # ---------- 6. CBF (predict_features, Stage 1) ----------
    logger.info("=== Tahap 6: Content-Based Filtering & Clustering ===")
    full_df_for_items = pd.concat([train_df, val_df, test_df], ignore_index=True)
    cbf_config = CBFConfig(
        method=config["cbf_clustering"]["method"],
        k_min=2,
        k_max=20,
        pca_components=config["cbf_clustering"].get("pca_components", 50),
        random_state=exp_cfg["seed"],
    )
    cbf_predictor = CBFPredictor(cbf_config=cbf_config)
    cbf_predictor.fit(full_df_for_items, train_df)
    logger.info(
        "CBF clustering selesai: K optimal=%d (metode=%s)", cbf_predictor.clusterer.best_k, cbf_config.method
    )

    cbf_scalar = {}
    cbf_features = {}
    for name, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cbf_scalar[name] = cbf_predictor.predict(part_df, rating_scale)
        cbf_features[name] = cbf_predictor.predict_features(part_df)

    # ---------- 7. Fusion (DIGANTI sesuai skenario) ----------
    logger.info("=== Tahap 7: Fusion (skenario '%s') ===", scenario)
    n_params = None
    gate_weights_test = None
    gate_modalities = None  # modalitas aktual yg dipakai AGF (utk nama kolom gate CSV)

    if scenario == "static_pyabsa":
        # Sel faktorial "Static (NMF+DT) + Model-based ABSA" -- reuse
        # fusi Fase 1 APA ADANYA, cuma sumber sentimen diganti PyABSA.
        # NMFDecisionTreeFusion sudah generik terima input 2D (lihat
        # fusion_nmf_dt.py), jadi vektor ABSA 5-dim diterima langsung.
        fusion_config = FusionConfig(
            nmf_components=config["fusion_baseline"]["nmf_components"],
            dt_max_depth=config["fusion_baseline"]["dt_max_depth"],
            random_state=exp_cfg["seed"],
        )
        fusion_model = NMFDecisionTreeFusion(fusion_config)
        t0 = time.time()
        fusion_model.fit(
            sentiment_scores=absa_features["train"],
            deepmf_preds=deepmf_scalar["train"],
            cbf_preds=cbf_scalar["train"],
            y_true_ratings=train_df["stars"].values,
        )
        train_time = time.time() - t0
        t0 = time.time()
        test_final_preds = fusion_model.predict(
            sentiment_scores=absa_features["test"], deepmf_preds=deepmf_scalar["test"], cbf_preds=cbf_scalar["test"]
        )
        predict_time = time.time() - t0
        test_final_preds = np.clip(test_final_preds, rating_scale[0], rating_scale[1])

    elif scenario == "weighted_avg":
        # Baseline eksternal Tier 2 "Weighted-average tetap" -- gating
        # NAIF tanpa jaringan neural sama sekali, beroperasi di ruang
        # SKALAR (bukan vektor spt AGF): regresi linear sederhana atas 3
        # prediksi skalar (DeepMF, CBF, ringkasan ABSA) -- pembanding
        # paling sederhana yang mungkin utk "gating".
        absa_scalar_train = absa_features["train"][:, ABSA_VECTOR_FEATURE_NAMES.index("mean_positive_prob")]
        absa_scalar_test = absa_features["test"][:, ABSA_VECTOR_FEATURE_NAMES.index("mean_positive_prob")]
        X_train = np.column_stack([deepmf_scalar["train"], cbf_scalar["train"], absa_scalar_train * 4 + 1])
        X_test = np.column_stack([deepmf_scalar["test"], cbf_scalar["test"], absa_scalar_test * 4 + 1])
        t0 = time.time()
        lr = LinearRegression()
        lr.fit(X_train, train_df["stars"].values)
        train_time = time.time() - t0
        t0 = time.time()
        test_final_preds = np.clip(lr.predict(X_test), rating_scale[0], rating_scale[1])
        predict_time = time.time() - t0
        n_params = X_train.shape[1] + 1  # bobot + intercept

    elif scenario == "static_keyword_pyabsa":
        # KONTROL ATRIBUSI: tree NMF+DT (sama spt A2-IRM) TAPI sentiment =
        # [keyword ABSA (concat+conf) HSTACK PyABSA-rich]. deepmf/cbf skalar
        # sama. Kalau ini mengalahkan A2-IRM sebanyak AGF+pyabsa -> tree juga
        # bisa pakai PyABSA (AGF bukan mekanisme unik). NMFDecisionTreeFusion
        # sudah generik terima sentiment 2D multi-kolom.
        splits_for_pyabsa = {"train": train_df, "val": val_df, "test": test_df}
        pyabsa_rich = _compute_pyabsa_rich_modality(config, exp_cfg, splits_for_pyabsa, rich=True)
        sent = {s: np.hstack([absa_features[s], pyabsa_rich[s]]) for s in ("train", "test")}
        fusion_config = FusionConfig(
            nmf_components=config["fusion_baseline"]["nmf_components"],
            dt_max_depth=config["fusion_baseline"]["dt_max_depth"],
            random_state=exp_cfg["seed"],
        )
        fusion_model = NMFDecisionTreeFusion(fusion_config)
        t0 = time.time()
        fusion_model.fit(
            sentiment_scores=sent["train"], deepmf_preds=deepmf_scalar["train"],
            cbf_preds=cbf_scalar["train"], y_true_ratings=train_df["stars"].values,
        )
        train_time = time.time() - t0
        t0 = time.time()
        test_final_preds = np.clip(
            fusion_model.predict(sent["test"], deepmf_scalar["test"], cbf_scalar["test"]),
            rating_scale[0], rating_scale[1],
        )
        predict_time = time.time() - t0
        logger.info("KONTROL ATRIBUSI: tree NMF+DT([keyword ABSA + PyABSA-rich, deepmf, cbf]).")

    else:
        # Semua varian AGF (Full/Attention-only/Gating-only/leave-one-out/
        # Concat+MLP/AGF+Keyword) -- SATU class, dibedakan kombinasi
        # modalitas aktif + 2 flag config (lihat AGF_SCENARIOS di atas).
        scenario_cfg = AGF_SCENARIOS[scenario]
        modalities = scenario_cfg["modalities"]

        all_feature_sources = {"deepmf": deepmf_latent, "cbf": cbf_features, "absa": absa_features}

        rating_min, rating_max = rating_scale
        scale_range = rating_max - rating_min

        # REPRESENTASI ASIMETRIS (redesign Gap 2): DeepMF/CBF diganti PREDIKSI
        # SKALAR ternormalisasi (bukan laten/fitur mentah) -- keduanya
        # prediktor rating, sinyal terkuatnya = output skalarnya. ABSA tetap
        # vektor kaya (tidak diubah). Menjawab temuan leave_out_deepmf~=full_agf
        # (laten DeepMF nyaris tak berkontribusi).
        if representation == "asymmetric":
            for mod, scalar_dict in {"deepmf": deepmf_scalar, "cbf": cbf_scalar}.items():
                all_feature_sources[mod] = {
                    split: ((scalar_dict[split].reshape(-1, 1) - rating_min) / scale_range).astype(np.float32)
                    for split in ("train", "val", "test")
                }
            logger.info("REPRESENTASI ASIMETRIS AKTIF: DeepMF/CBF -> prediksi skalar ternormalisasi.")

        # MODALITAS EKSTRA PyABSA per-aspek: sinyal BARU utk AGF yg tree
        # A2-IRM (base) TIDAK punya -- SENGAJA tidak dimasukkan ke base,
        # supaya kalau AGF mengoreksi base ke bawah A2-IRM, itu jelas berkat
        # sinyal PyABSA+attention (bukan re-fit fitur yg sama).
        aspect_seq = None  # Jalur X: sequence aspek utk AspectSequencePooling
        aspect_vocab = None
        if extra_pyabsa in ("rich", "summary"):
            splits_for_pyabsa = {"train": train_df, "val": val_df, "test": test_df}
            pyabsa_extra = _compute_pyabsa_rich_modality(
                config, exp_cfg, splits_for_pyabsa, rich=(extra_pyabsa == "rich")
            )
            all_feature_sources["pyabsa"] = pyabsa_extra
            if "pyabsa" not in modalities:
                modalities = modalities + ["pyabsa"]
        elif extra_pyabsa == "perseq":
            splits_for_pyabsa = {"train": train_df, "val": val_df, "test": test_df}
            aspect_vocab, aspect_seq = _compute_pyabsa_aspect_sequences(
                config, exp_cfg, splits_for_pyabsa
            )

        # RESIDUAL base: base = prediksi NMF+DecisionTree (sama spt A2-IRM)
        # atas [absa, deepmf_scalar, cbf_scalar]; AGF belajar koreksi di
        # atasnya. base pakai KETIGA sinyal terlepas modalitas aktif skenario.
        #
        # "static_fusion" (lama, CACAT): base train IN-FOLD -- DecisionTree
        #   fit train ~sempurna -> residual train ~0 -> AGF tak belajar apa2
        #   (bug stacking klasik). Disimpan HANYA utk perbandingan.
        # "static_fusion_oof" (BENAR): base train OUT-OF-FOLD (5-fold CV) --
        #   tiap sampel train diprediksi base yg TIDAK dilatih dgnnya ->
        #   residual train NYATA -> AGF punya sinyal utk dipelajari & bisa
        #   transfer ke test. val/test tetap dari base fit-seluruh-train
        #   (sudah out-of-sample).
        base_norm = None
        if residual_base in ("static_fusion", "static_fusion_oof"):
            fusion_config = FusionConfig(
                nmf_components=config["fusion_baseline"]["nmf_components"],
                dt_max_depth=config["fusion_baseline"]["dt_max_depth"],
                random_state=exp_cfg["seed"],
            )
            base_norm = {}
            y_train = train_df["stars"].values

            if residual_base == "static_fusion_oof":
                from sklearn.model_selection import KFold
                n_folds = 5
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=exp_cfg["seed"])
                oof_train = np.zeros(len(train_df), dtype=np.float64)
                for tr_idx, oof_idx in kf.split(np.arange(len(train_df))):
                    fold_fusion = NMFDecisionTreeFusion(fusion_config)
                    fold_fusion.fit(
                        sentiment_scores=absa_features["train"][tr_idx],
                        deepmf_preds=deepmf_scalar["train"][tr_idx],
                        cbf_preds=cbf_scalar["train"][tr_idx],
                        y_true_ratings=y_train[tr_idx],
                    )
                    oof_train[oof_idx] = fold_fusion.predict(
                        sentiment_scores=absa_features["train"][oof_idx],
                        deepmf_preds=deepmf_scalar["train"][oof_idx],
                        cbf_preds=cbf_scalar["train"][oof_idx],
                    )
                base_norm["train"] = (
                    (np.clip(oof_train, rating_min, rating_max) - rating_min) / scale_range
                ).astype(np.float32)
                oof_rmse = float(np.sqrt(np.mean(
                    (np.clip(oof_train, rating_min, rating_max) - y_train) ** 2)))
                logger.info(
                    "RESIDUAL OOF: base train %d-fold OOF RMSE=%.4f (residual train NYATA, bukan ~0).",
                    n_folds, oof_rmse,
                )

            # base fit di SELURUH train -> prediksi val & test (out-of-sample),
            # dan train juga kalau mode in-fold lama.
            base_fusion = NMFDecisionTreeFusion(fusion_config)
            base_fusion.fit(
                sentiment_scores=absa_features["train"],
                deepmf_preds=deepmf_scalar["train"],
                cbf_preds=cbf_scalar["train"],
                y_true_ratings=y_train,
            )
            for split in ("train", "val", "test"):
                if split == "train" and residual_base == "static_fusion_oof":
                    continue  # train sudah diisi OOF di atas
                base_pred = base_fusion.predict(
                    sentiment_scores=absa_features[split],
                    deepmf_preds=deepmf_scalar[split],
                    cbf_preds=cbf_scalar[split],
                )
                base_norm[split] = (
                    (np.clip(base_pred, rating_min, rating_max) - rating_min) / scale_range
                ).astype(np.float32)
            base_test_rmse = float(np.sqrt(np.mean(
                (np.clip(base_fusion.predict(absa_features["test"], deepmf_scalar["test"], cbf_scalar["test"]),
                         rating_min, rating_max) - test_df["stars"].values) ** 2)))
            logger.info(
                "RESIDUAL base=%s AKTIF: base NMF+DT (RMSE test base=%.4f), AGF belajar koreksi.",
                residual_base, base_test_rmse,
            )

        # DIAGNOSTIK 1 (use_scalar_preds): tambahkan prediksi rating skalar
        # DeepMF/CBF (yg SUDAH terkalibrasi) sbg 1 kolom fitur ekstra di
        # modalitas masing-masing. AGF versi asli cuma pakai laten/fitur
        # mentah -- diuji apakah mengembalikan sinyal prediksi yg dibuang.
        if use_scalar_preds:
            scalar_sources = {"deepmf": deepmf_scalar, "cbf": cbf_scalar}
            for mod, scalar_dict in scalar_sources.items():
                if mod in all_feature_sources:
                    all_feature_sources[mod] = {
                        split: np.hstack([all_feature_sources[mod][split], scalar_dict[split].reshape(-1, 1)])
                        for split in ("train", "val", "test")
                    }
            logger.info("DIAGNOSTIK use_scalar_preds AKTIF: prediksi skalar DeepMF/CBF ditambah sbg fitur.")

        # DIAGNOSTIK 2 (input_standardize): StandardScaler per-modalitas, fit
        # HANYA di train, apply ke semua split -- menyamakan skala antar-
        # modalitas yg sangat berbeda (laten DeepMF vs PCA CBF vs ABSA [0,1]).
        if input_standardize:
            for mod in list(all_feature_sources.keys()):
                if mod not in modalities:
                    continue
                scaler = StandardScaler()
                scaler.fit(all_feature_sources[mod]["train"])
                all_feature_sources[mod] = {
                    split: scaler.transform(all_feature_sources[mod][split]).astype(np.float32)
                    for split in ("train", "val", "test")
                }
            logger.info("DIAGNOSTIK input_standardize AKTIF: StandardScaler per-modalitas (fit di train).")

        modality_dims = {m: all_feature_sources[m]["train"].shape[1] for m in modalities}

        def build_features(split_name: str) -> dict[str, np.ndarray]:
            return {m: all_feature_sources[m][split_name] for m in modalities}

        train_y_norm = ((train_df["stars"].values - rating_min) / scale_range).astype(np.float32)
        val_y_norm = ((val_df["stars"].values - rating_min) / scale_range).astype(np.float32)

        agf_cfg = AGFConfig(
            d=config.get("agf", {}).get("d", 64),
            n_heads=config.get("agf", {}).get("n_heads", 2),
            epochs=config.get("agf", {}).get("epochs", 30),
            batch_size=config.get("agf", {}).get("batch_size", 512),
            learning_rate=config.get("agf", {}).get("learning_rate", 0.001),
            weight_decay=config.get("agf", {}).get("weight_decay", 0.0),
            use_attention=scenario_cfg["use_attention"],
            pooling=scenario_cfg["pooling"],
            residual=(residual_base != "none"),
            aspect_pooling=(extra_pyabsa == "perseq"),
            aspect_vocab_size=(len(aspect_vocab) if aspect_vocab is not None else 0),
            aspect_emb_dim=config.get("agf", {}).get("aspect_emb_dim", 16),
        )
        agf_trainer = AttentionGatedFusionTrainer(modality_dims, agf_cfg)
        train_time = agf_trainer.fit(
            build_features("train"), train_y_norm, build_features("val"), val_y_norm,
            train_base_norm=(base_norm["train"] if base_norm else None),
            val_base_norm=(base_norm["val"] if base_norm else None),
            train_aspect_seq=(aspect_seq["train"] if aspect_seq else None),
            val_aspect_seq=(aspect_seq["val"] if aspect_seq else None),
        )
        n_params = agf_trainer.n_parameters
        # modalitas aktual utk nama kolom gate (+ token aspek kalau perseq)
        gate_modalities = modalities + (["pyabsa_aspect"] if extra_pyabsa == "perseq" else [])

        t0 = time.time()
        test_final_preds, gate_weights_test = agf_trainer.predict(
            build_features("test"), rating_scale,
            base_norm=(base_norm["test"] if base_norm else None),
            aspect_seq=(aspect_seq["test"] if aspect_seq else None),
        )
        predict_time = time.time() - t0
        test_final_preds = np.clip(test_final_preds, rating_scale[0], rating_scale[1])

    # ---------- 8. Evaluasi akhir (verbatim, reuse infra Fase 1) ----------
    logger.info("=== Tahap 8: Evaluasi akhir (test set held-out) ===")
    y_true = test_df["stars"].values
    rmse, mae = compute_rmse_mae(y_true, test_final_preds)
    sanity_check_rmse(rmse, rating_scale)

    logger.info("=" * 60)
    logger.info("HASIL AGF (domain: %s, skenario: %s)", exp_cfg["domain"], scenario)
    logger.info("RMSE: %.4f | MAE: %.4f | train_time: %.1fs | predict_time: %.3fs | n_param: %s",
                rmse, mae, train_time, predict_time, n_params)
    logger.info("=" * 60)

    test_df_eval = test_df.copy()
    test_df_eval["pred_score"] = test_final_preds
    relevance_threshold = 4.0
    ranked_items_per_user, relevant_items_per_user = {}, {}
    for user_id, group in test_df_eval.groupby("user_id"):
        ranked_items_per_user[user_id] = group.sort_values("pred_score", ascending=False)["business_id"].tolist()
        relevant_items_per_user[user_id] = set(group[group["stars"] >= relevance_threshold]["business_id"])

    k_values = config["evaluation"]["k_values"]
    precision_k, recall_k, ndcg_k = precision_recall_ndcg_at_k(ranked_items_per_user, relevant_items_per_user, k_values)
    logger.info("Precision@K: %s | Recall@K: %s | NDCG@K: %s", precision_k, recall_k, ndcg_k)

    results_dir = Path(config["logging"]["checkpoint_dir"]).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # run_tag disisipkan ke prefix supaya run diagnostik TIDAK menimpa hasil
    # utama 150-run (agf_{scenario}_{domain}_seed{seed}.yaml).
    tag_suffix = f"_{run_tag}" if run_tag else ""
    results_prefix = f"agf_{scenario}{tag_suffix}"
    model_name = f"a2fusionrs_agf_{scenario}{tag_suffix}"
    results_path = results_dir / f"{results_prefix}_{exp_cfg['domain']}_seed{exp_cfg['seed']}.yaml"

    results_summary = {
        "model_name": model_name,
        "domain": exp_cfg["domain"],
        "seed": exp_cfg["seed"],
        "scenario": scenario,
        "absa_source": absa_source,
        "n_test_samples": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "precision_at_k": {int(k): v for k, v in precision_k.items()},
        "recall_at_k": {int(k): v for k, v in recall_k.items()},
        "ndcg_at_k": {int(k): v for k, v in ndcg_k.items()},
        # Instrumentasi efisiensi (Tier 3, dicatat sejak awal -- lihat
        # attention_gated_fusion_design.md Bagian 3 poin 8).
        "train_time_seconds": train_time,
        "predict_time_seconds": predict_time,
        "n_parameters": n_params,
        "input_standardize": input_standardize,
        "use_scalar_preds": use_scalar_preds,
        "representation": representation,
        "residual_base": residual_base,
        "extra_pyabsa": extra_pyabsa,
        "run_tag": run_tag,
        "notes": f"A2-FusionRS Fase 2, skenario ablasi '{scenario}', sumber ABSA '{absa_source}'"
        + (f" [DIAGNOSTIK: {run_tag}]" if run_tag else "."),
    }
    save_results_yaml(results_path, results_summary, config=config)

    predictions_path = results_dir / f"predictions_{results_path.stem}.csv"
    save_predictions(predictions_path, test_df, test_final_preds)

    # Bobot gate per-baris (Tier 3, interpretability) -- HANYA ada kalau
    # skenario memakai pooling="gate".
    if gate_weights_test is not None:
        scenario_cfg = AGF_SCENARIOS[scenario]
        gate_cols = gate_modalities if gate_modalities is not None else scenario_cfg["modalities"]
        gate_df = pd.DataFrame(gate_weights_test, columns=[f"gate_{m}" for m in gate_cols])
        gate_df.insert(0, "review_id", test_df["review_id"].values)
        gate_path = results_dir / f"gates_{results_path.stem}.csv"
        gate_df.to_csv(gate_path, index=False)
        logger.info("Bobot gate per-baris disimpan ke %s.", gate_path)

    # §6.5 interpretability (Exp-B/C) -- HANYA perseq (AspectSequencePooling aktif).
    if export_interpretability:
        if extra_pyabsa != "perseq":
            logger.warning("--export-interpretability diabaikan: hanya berlaku utk --extra-pyabsa perseq.")
        else:
            _export_interpretability(
                agf_trainer, build_features("test"),
                base_norm["test"] if base_norm else None,
                aspect_seq["test"], aspect_vocab, test_df, test_final_preds,
                rating_scale, results_dir, results_path.stem,
            )

    logger.info("Pipeline AGF SELESAI. RMSE=%.4f (skenario '%s', domain '%s').", rmse, scenario, exp_cfg["domain"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan pipeline Attention-Gated Fusion (A2-FusionRS Fase 2)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True, choices=ALL_SCENARIOS)
    parser.add_argument(
        "--seed", type=int, default=None, help="Override experiment.seed dari config -- utk protokol multi-seed."
    )
    parser.add_argument(
        "--input-standardize", action="store_true",
        help="DIAGNOSTIK Stage 7: StandardScaler per-modalitas sebelum AGF (default off = perilaku 150-run asli).",
    )
    parser.add_argument(
        "--use-scalar-preds", action="store_true",
        help="DIAGNOSTIK Stage 7: umpankan prediksi skalar DeepMF/CBF sbg fitur tambahan (default off).",
    )
    parser.add_argument(
        "--representation", type=str, default="vector", choices=["vector", "asymmetric"],
        help="REDESIGN Gap 2: 'asymmetric' = DeepMF/CBF pakai prediksi skalar (bukan laten/fitur), ABSA tetap vektor.",
    )
    parser.add_argument(
        "--residual-base", type=str, default="none",
        choices=["none", "static_fusion", "static_fusion_oof"],
        help="REDESIGN Gap 2: AGF belajar KOREKSI di atas base NMF+DT. 'static_fusion_oof' = base train "
        "out-of-fold (stacking BENAR); 'static_fusion' = in-fold (cacat, utk perbandingan).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="Override agf.weight_decay (L2 Adam) -- utk residual, menekan koreksi ke robust-only.",
    )
    parser.add_argument(
        "--extra-pyabsa", type=str, default="none", choices=["none", "rich", "summary", "perseq"],
        help="Tambah sinyal PyABSA per-aspek ke AGF (BUKAN ke base): 'rich'=order-stats kontras (fixed, "
        "tree juga bisa), 'summary'=5-dim rata-rata, 'perseq'=Jalur X sequence aspek + IDENTITAS via "
        "AspectSequencePooling (tree TAK BISA -- keunggulan struktural AGF).",
    )
    parser.add_argument(
        "--run-tag", type=str, default="",
        help="Suffix nama file hasil supaya run diagnostik tidak menimpa hasil utama 150-run (mis. 'norm').",
    )
    parser.add_argument(
        "--export-interpretability", action="store_true",
        help="§6.5: ekspor atensi per-aspek (studi kasus) + uji faithfulness perturbasi. "
        "HANYA berlaku utk --extra-pyabsa perseq (AspectSequencePooling).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.weight_decay is not None:
        cfg.setdefault("agf", {})["weight_decay"] = args.weight_decay
    run_pipeline(
        cfg, args.scenario,
        input_standardize=args.input_standardize,
        use_scalar_preds=args.use_scalar_preds,
        representation=args.representation,
        residual_base=args.residual_base,
        extra_pyabsa=args.extra_pyabsa,
        run_tag=args.run_tag,
        export_interpretability=args.export_interpretability,
    )
