"""
src/a2fusionrs/pyabsa_scorer.py

Aspect-Based Sentiment Analysis (ABSA) BERBASIS MODEL (PyABSA ATEPC), untuk
Fase 2 A2-FusionRS -- menggantikan `absa_bert.py` (keyword-matching, Fase 1)
sebagai sumber sinyal aspek untuk stream ABSA di Attention-Gated Fusion.
Lihat `phase2_notes/pyabsa_investigation.md` untuk hasil benchmark & alasan
pemilihan checkpoint "english" (bukan "multilingual" -- 4x cakupan, 4,6x
lebih banyak aspek/review, lihat Bagian 2 catatan tsb).

TANGGUNG JAWAB modul ini TERBATAS pada skoring mentah + cache per-review
(Stage 0 rencana implementasi) -- KONVERSI ke vektor fixed-size untuk
input Attention-Gated Fusion SENGAJA TIDAK dilakukan di sini, itu tanggung
jawab `run_attention_gated_fusion.py` (Stage 3). Alasan: PyABSA menghasilkan
istilah aspek OPEN-VOCABULARY (beda-beda tiap review, bukan dari taksonomi
tetap 4-6 kategori seperti keyword-matching) -- strategi pemetaan ke vektor
tetap adalah keputusan desain fusion, bukan keputusan scoring, dan sengaja
dipisah supaya modul ini tetap reusable apa pun strategi vektorisasi yang
akhirnya dipilih (lihat "Isu desain terbuka" di pyabsa_investigation.md).

Skor TIDAK bergantung pada split train/val/test atau seed eksperimen --
1 review menghasilkan skor yang sama terlepas dari split mana dia jatuh.
Karena itu skoring dilakukan SEKALI per domain atas SELURUH review (lihat
`run_pyabsa_scoring.py`), di-cache per `review_id`, dan dipakai ulang untuk
semua seed & skenario ablasi -- ini yang membuat biaya ~14,6 jam GPU (3
domain) jadi biaya TETAP, bukan dikalikan jumlah seed/skenario (lihat
estimasi biaya di attention_gated_fusion_design.md Bagian 4).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PyABSAConfig:
    # "english" direkomendasikan utk domain mayoritas berbahasa Inggris
    # (restoran, Amazon Electronics, TripAdvisor Hotel -- SEMUA domain
    # proyek ini) -- lihat rekomendasi checkpoint di pyabsa_investigation.md.
    # "multilingual" TERSEDIA tapi TIDAK direkomendasikan (cakupan jauh lebih
    # rendah: 17,2% vs 70,0% pada sample benchmark yang sama).
    checkpoint: str = "english"
    # None -> biarkan PyABSA auto-detect (pakai GPU kalau ada, mis. di Colab
    # dgn runtime GPU). Set eksplisit False utk paksa CPU (mis. smoke test
    # lokal tanpa GPU dedicated).
    auto_device: bool | None = None


class PyABSAAspectScorer:
    """Wrapper checkpoint ATEPC pretrained PyABSA -- TIDAK ada training/
    fine-tuning model di kelas ini, murni inferensi checkpoint off-the-shelf
    (lihat rekomendasi & keterbatasan yang didokumentasikan di
    phase2_notes/pyabsa_investigation.md sebelum memutuskan pakai kelas ini
    apa adanya vs fine-tune model ABSA sendiri)."""

    def __init__(self, config: PyABSAConfig | None = None):
        self.config = config or PyABSAConfig()
        logger.info(
            "Memuat checkpoint PyABSA ATEPC '%s' (auto_device=%s) -- proses "
            "ini bisa makan waktu signifikan pada panggilan pertama (unduh/"
            "load checkpoint dari disk).",
            self.config.checkpoint,
            self.config.auto_device,
        )
        t0 = time.time()
        # Import di dalam __init__ (bukan top-level module) -- pyabsa &
        # dependensinya (spacy, findfile, dll, lihat
        # requirements-phase2-pyabsa.txt) TIDAK dibutuhkan sama sekali untuk
        # menjalankan pipeline Fase 1 (A2-IRM), jadi tidak dipaksa ter-import
        # setiap kali modul lain di package a2fusionrs di-import.
        from pyabsa import AspectTermExtraction as ATEPC

        kwargs = {}
        if self.config.auto_device is not None:
            kwargs["auto_device"] = self.config.auto_device
        self.extractor = ATEPC.AspectExtractor(checkpoint=self.config.checkpoint, **kwargs)
        logger.info("Checkpoint PyABSA dimuat dalam %.1f detik.", time.time() - t0)

    def score_dataframe(
        self, df: pd.DataFrame, text_column: str = "text_bert", review_id_column: str = "review_id"
    ) -> pd.DataFrame:
        """Jalankan ekstraksi aspek+sentimen mentah PyABSA pada SELURUH baris
        df, SATU kali panggilan `predict()` (bukan loop per-baris) untuk
        efisiensi -- kembalikan 1 baris per review dgn daftar aspek/sentimen/
        confidence MENTAH (list, belum divektorkan/diagregasi).

        Kolom hasil:
        - review_id
        - n_aspects: jumlah istilah aspek yang ditemukan PyABSA (0 = review
          ini tidak menghasilkan aspek sama sekali -- BEDA dari
          keyword-matching yang selalu punya fallback whole-review; PyABSA
          off-the-shelf TIDAK menjamin minimal 1 aspek per review, lihat
          catatan cakupan di pyabsa_investigation.md).
        - aspects_json / sentiments_json / confidences_json / probs_json:
          list mentah hasil PyABSA, di-serialize JSON supaya aman disimpan
          ke CSV (kolom list Python tidak native didukung format CSV).
          `probs_json` menyimpan distribusi probabilitas 3-kelas penuh
          (urutan [negative, neutral, positive] per aspek) -- lebih kaya
          dari `sentiments`/`confidences` (label top-1 + confidence-nya
          saja), disimpan berjaga-jaga supaya Stage 3 (vektorisasi utk
          fusion) punya opsi memakai distribusi penuh tanpa perlu
          menjalankan ulang inferensi PyABSA yang mahal (~14,6 jam GPU).
        """
        texts = df[text_column].fillna("").tolist()
        review_ids = df[review_id_column].tolist()

        logger.info("PyABSA: menjalankan predict() pada %d review...", len(texts))
        t0 = time.time()
        results = self.extractor.predict(
            texts, print_result=False, save_result=False, pred_sentiment=True, ignore_error=True
        )
        elapsed = time.time() - t0
        logger.info(
            "PyABSA: selesai dalam %.1f menit (%.3f detik/review).",
            elapsed / 60,
            elapsed / len(texts) if texts else 0.0,
        )

        rows = []
        for review_id, res in zip(review_ids, results):
            if res is None:
                # predict() dgn ignore_error=True mengembalikan None utk
                # baris yang gagal diproses (mis. teks kosong/rusak) alih-alih
                # meledakkan seluruh batch -- diperlakukan sbg 0 aspek.
                rows.append(
                    {
                        "review_id": review_id,
                        "n_aspects": 0,
                        "aspects_json": "[]",
                        "sentiments_json": "[]",
                        "confidences_json": "[]",
                        "probs_json": "[]",
                    }
                )
                continue
            aspects = res.get("aspect", [])
            sentiments = res.get("sentiment", [])
            confidences = [float(c) for c in res.get("confidence", [])]
            probs = [[float(p) for p in triplet] for triplet in res.get("probs", [])]
            rows.append(
                {
                    "review_id": review_id,
                    "n_aspects": len(aspects),
                    "aspects_json": json.dumps(aspects),
                    "sentiments_json": json.dumps(sentiments),
                    "confidences_json": json.dumps(confidences),
                    "probs_json": json.dumps(probs),
                }
            )
        return pd.DataFrame(rows)

    def coverage_report(self, scored_df: pd.DataFrame) -> dict:
        """Diagnostik cakupan (WAJIB dilog & disimpan tiap run, sama seperti
        `aspect_coverage_report()` di absa_bert.py) -- dipakai jg utk
        verifikasi Stage 0: hasil pada sample 500-review harus cocok dgn
        benchmark yg sudah terdokumentasi (70,0% cakupan, 2,77 rata-rata
        aspek/review utk checkpoint 'english', lihat pyabsa_investigation.md)
        sebelum lanjut ke run skala penuh 3-domain.
        """
        total = len(scored_df)
        n_with_aspects = int((scored_df["n_aspects"] > 0).sum())
        return {
            "n_reviews": total,
            "n_with_any_aspect": n_with_aspects,
            "pct_with_any_aspect": n_with_aspects / total if total else 0.0,
            "avg_aspects_per_review": float(scored_df["n_aspects"].mean()) if total else 0.0,
        }


def load_cached_scores(cache_path: str) -> pd.DataFrame:
    """Muat cache skor PyABSA (`aspects_json`/`sentiments_json`/
    `confidences_json` sbg string) & decode kembali jadi list Python --
    dipakai `run_attention_gated_fusion.py` (Stage 3) supaya tidak
    mengulang parsing JSON manual di banyak tempat."""
    df = pd.read_csv(cache_path)
    for col in ("aspects_json", "sentiments_json", "confidences_json", "probs_json"):
        decoded_col = col.replace("_json", "")
        df[decoded_col] = df[col].apply(json.loads)
    return df


ABSA_VECTOR_FEATURE_NAMES = [
    "n_aspects_norm",
    "mean_positive_prob",
    "mean_negative_prob",
    "mean_confidence",
    "std_positive_prob",
]


def vectorize_absa_features(
    scored_df: pd.DataFrame,
    fallback_scores: dict | None = None,
    evidence_cap: int = 3,
) -> np.ndarray:
    """Ubah hasil skor PyABSA mentah (kolom 'aspects'/'sentiments'/
    'confidences'/'probs' berisi list Python -- hasil `load_cached_scores()`
    atau `PyABSAAspectScorer.score_dataframe()` langsung) jadi vektor
    fixed-size per baris, untuk dikonsumsi Attention-Gated Fusion sebagai
    modalitas 'absa'. KEPUTUSAN DESAIN (Stage 3 rencana implementasi,
    lihat phase2_notes/attention_gated_fusion_design.md & catatan "Isu
    desain terbuka" di pyabsa_investigation.md):

    PyABSA open-vocabulary (jumlah & istilah aspek beda-beda tiap review)
    -- TIDAK dipaksa dipetakan ke taksonomi keyword tetap 4-6 kategori
    (itu akan mengulang masalah averaging yang sudah terbukti merusak
    sinyal di varian "mean" Fase 1). Sebagai gantinya, dipakai vektor
    ringkasan (summary statistics) atas SEMUA aspek yang ditemukan PyABSA
    per baris:

    1. n_aspects_norm: jumlah aspek ternormalisasi min(n/evidence_cap, 1)
       -- proxy "seberapa banyak bukti tekstual", pola SAMA dengan
       `_evidence_confidence()` di absa_bert.py (evidence_cap default 3,
       konsisten dengan cap yang sama di sana).
    2. mean_positive_prob / mean_negative_prob: rata-rata P(positive) dan
       P(negative) (dari `probs`, urutan [negative, neutral, positive])
       lintas aspek yang ditemukan -- lebih kaya dari sekadar label top-1.
    3. mean_confidence: rata-rata confidence (probabilitas kelas top-1)
       lintas aspek.
    4. std_positive_prob: std P(positive) lintas aspek -- proxy "seberapa
       campur/heterogen sentimen antar aspek dalam 1 review" (mis. "kamar
       bagus tapi servis buruk" -> std tinggi); 0 kalau cuma 1 aspek atau
       0 aspek ditemukan.

    Baris dengan 0 aspek terdeteksi (PyABSA TIDAK menjamin cakupan seperti
    keyword-matching yang selalu punya fallback whole-review -- lihat
    benchmark cakupan 70% checkpoint "english" di pyabsa_investigation.md)
    diisi dari `fallback_scores` (dict `review_id -> skor [0,1]`) kalau
    diberikan -- dimaksudkan untuk diisi skor `GlobalSentimentBERT` yang
    SUDAH di-checkpoint dari Fase 1 (`run_baseline.py`), dipanggil HANYA
    untuk baris yang PyABSA gagal temukan aspek (~30% baris, bukan
    seluruh dataset) -- reuse checkpoint, TIDAK ada training/inferensi
    tambahan yang mahal. Ini menjaga PARALEL metodologis dengan ABSA
    keyword-based Fase 1 (yang juga fallback ke skor whole-review saat
    tidak ada aspek match), supaya perbandingan faktorial Fusion x
    ABSA-extraction (lihat design doc) tidak bias oleh kebijakan fallback
    yang berbeda antara kedua metode ekstraksi.

    Kalau `fallback_scores` TIDAK diberikan (mis. smoke test cepat), baris
    0-aspek diisi netral (0,5/0,5/0/0/0) -- BUKAN NaN, supaya training AGF
    tidak pernah menerima NaN dari modalitas ini.

    Return: array (n_baris, 5), urutan kolom = `ABSA_VECTOR_FEATURE_NAMES`.
    """
    n = len(scored_df)
    out = np.zeros((n, len(ABSA_VECTOR_FEATURE_NAMES)), dtype=np.float32)

    for row_idx, row in enumerate(scored_df.itertuples(index=False)):
        n_aspects = row.n_aspects
        probs = row.probs  # list of [neg, neu, pos] per aspek
        confidences = row.confidences

        if n_aspects == 0 or not probs:
            if fallback_scores is not None:
                fallback = fallback_scores.get(row.review_id, 0.5)
            else:
                fallback = 0.5
            out[row_idx] = [0.0, fallback, 1.0 - fallback, 0.0, 0.0]
            continue

        pos_probs = np.array([p[2] for p in probs], dtype=np.float32)
        neg_probs = np.array([p[0] for p in probs], dtype=np.float32)
        conf_arr = np.array(confidences, dtype=np.float32)

        out[row_idx] = [
            min(n_aspects / evidence_cap, 1.0),
            float(pos_probs.mean()),
            float(neg_probs.mean()),
            float(conf_arr.mean()) if len(conf_arr) else 0.0,
            float(pos_probs.std()) if len(pos_probs) > 1 else 0.0,
        ]

    return out


ABSA_RICH_FEATURE_NAMES = [
    "n_aspects_norm",        # jumlah aspek (bukti tekstual)
    "mean_positive_prob",    # rata-rata P(pos)
    "min_positive_prob",     # aspek PALING NEGATIF (kontras!)
    "max_positive_prob",     # aspek PALING POSITIF
    "range_positive_prob",   # spread = max-min (heterogenitas antar-aspek)
    "max_negative_prob",     # sinyal negatif terkuat
    "mean_confidence",       # rata-rata confidence
    "frac_negative_aspects", # fraksi aspek berlabel negatif (argmax=neg)
    "frac_positive_aspects", # fraksi aspek berlabel positif (argmax=pos)
]


def vectorize_absa_features_rich(
    scored_df: pd.DataFrame,
    fallback_scores: dict | None = None,
    evidence_cap: int = 3,
) -> np.ndarray:
    """Versi KAYA dari vectorize_absa_features: alih-alih hanya rata-rata
    (yang -- seperti varian 'Mean' A2-IRM -- MERUSAK kontras antar-aspek),
    fungsi ini mempertahankan STATISTIK URUTAN yang menangkap kontras:
    min/max/range P(pos) antar aspek, sinyal negatif terkuat, dan fraksi
    aspek negatif/positif.

    Motivasi (Stage 7+): tree A2-IRM ada di plafon utk fitur keyword-ABSA +
    DeepMF + CBF. Info PyABSA per-aspek (aspek open-vocabulary + sentimen
    individual) adalah sinyal BARU yg (a) keyword ABSA lewatkan, (b)
    averaging hancurkan, (c) tree tak bisa konsumsi tapi attention AGF bisa.
    Order-statistics ini versi fixed-size murah yg mempertahankan kontras --
    kalau ini sudah menambah sinyal ke koreksi residual AGF, full per-aspek
    attention-pooling (lebih mahal) baru dikerjakan.

    Contoh yg dibedakan dari mean: "produk bagus TAPI 1 aspek fatal negatif"
    -> min_positive_prob rendah, max_negative_prob tinggi, frac_negative>0 --
    semua ketangkap, sedangkan mean membaurkannya jadi 'agak positif'.

    Return: array (n, 9), urutan kolom = ABSA_RICH_FEATURE_NAMES.
    """
    n = len(scored_df)
    out = np.zeros((n, len(ABSA_RICH_FEATURE_NAMES)), dtype=np.float32)

    for row_idx, row in enumerate(scored_df.itertuples(index=False)):
        probs = row.probs  # list [neg, neu, pos] per aspek
        confidences = row.confidences

        if row.n_aspects == 0 or not probs:
            fb = (fallback_scores or {}).get(row.review_id, 0.5) if fallback_scores is not None else 0.5
            # mean/min/max_pos = fallback; kontras & fraksi = 0 (tidak ada aspek)
            out[row_idx] = [0.0, fb, fb, fb, 0.0, 1.0 - fb, 0.0, 0.0, 0.0]
            continue

        pos = np.array([p[2] for p in probs], dtype=np.float32)
        neg = np.array([p[0] for p in probs], dtype=np.float32)
        conf = np.array(confidences, dtype=np.float32)
        argmax = np.array([int(np.argmax(p)) for p in probs])  # 0=neg,1=neu,2=pos

        out[row_idx] = [
            min(row.n_aspects / evidence_cap, 1.0),
            float(pos.mean()),
            float(pos.min()),
            float(pos.max()),
            float(pos.max() - pos.min()),
            float(neg.max()),
            float(conf.mean()) if len(conf) else 0.0,
            float((argmax == 0).mean()),
            float((argmax == 2).mean()),
        ]

    return out


def build_aspect_vocab(scored_df: pd.DataFrame, top_k: int = 500) -> dict:
    """Bangun kosakata istilah aspek open-vocabulary dari kolom 'aspects'
    (list str) berdasar frekuensi -- ID 0 dicadangkan PAD, 1 UNK, sisanya
    top_k istilah tersering. Dipakai AspectSequencePooling (Jalur X) supaya
    AGF bisa membedakan IDENTITAS aspek ('baterai' vs 'layar'), sesuatu yg
    order-statistics buang dan tree tak bisa konsumsi.
    """
    from collections import Counter

    counter: Counter = Counter()
    for row in scored_df.itertuples(index=False):
        for a in row.aspects:
            t = str(a).lower().strip()
            if t:
                counter[t] += 1
    return {term: i + 2 for i, (term, _) in enumerate(counter.most_common(top_k))}


def build_aspect_sequences(
    scored_df: pd.DataFrame, vocab: dict, max_aspects: int = 8, fallback_scores: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ubah hasil PyABSA jadi sequence per-review (padded) utk
    AspectSequencePooling. Return (ids, feats, mask):
    - ids (n, max_aspects) int64: ID istilah aspek (0=PAD, 1=UNK)
    - feats (n, max_aspects, 4) float32: [P_neg, P_neu, P_pos, confidence]
    - mask (n, max_aspects) bool: True=aspek valid, False=padding

    Baris 0-aspek diberi 1 token sintetis dari fallback sentiment (UNK id,
    prob dari skor fallback) -- menjamin tiap baris punya >=1 token valid
    (cegah all-padding row yg bikin attention NaN).
    """
    n = len(scored_df)
    ids = np.zeros((n, max_aspects), dtype=np.int64)
    feats = np.zeros((n, max_aspects, 4), dtype=np.float32)
    mask = np.zeros((n, max_aspects), dtype=bool)

    for i, row in enumerate(scored_df.itertuples(index=False)):
        if row.n_aspects == 0 or not row.probs:
            fb = (fallback_scores or {}).get(row.review_id, 0.5) if fallback_scores is not None else 0.5
            ids[i, 0] = 1  # UNK
            feats[i, 0] = [1.0 - fb, 0.0, fb, abs(fb - 0.5) * 2.0]
            mask[i, 0] = True
            continue
        for j, (term, prob, conf) in enumerate(zip(row.aspects, row.probs, row.confidences)):
            if j >= max_aspects:
                break
            ids[i, j] = vocab.get(str(term).lower().strip(), 1)  # 1=UNK
            feats[i, j] = [float(prob[0]), float(prob[1]), float(prob[2]), float(conf)]
            mask[i, j] = True

    return ids, feats, mask


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton PyABSA scorer -- jalankan via run_pyabsa_scoring.py, "
        "jangan berdiri sendiri."
    )
