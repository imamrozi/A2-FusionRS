"""
src/a2fusionrs/absa_bert.py

Aspect-Based Sentiment Analysis (ABSA) RINGAN berbasis keyword, dipakai
sebagai varian ablasi Fase 1: mengganti stream sentimen GLOBAL baseline
Darraz et al. (satu skor per review) dengan skor ber-aspek (mis. food,
service, price, ambiance untuk domain restoran), TANPA training model baru
-- reuse model `GlobalSentimentBERT` yang SUDAH di-checkpoint dari
run_baseline.py, cuma diterapkan pada potongan kalimat per-aspek alih-alih
seluruh teks review.

DESAIN YANG DISENGAJA (konsisten dengan pendekatan di paper IEEE penulis
sebelumnya, "...Empirical Study on ABSA Quality Impact"): deteksi aspek
berbasis KEYWORD MATCHING sederhana, bukan model ABSA terlatih (tidak ada
dataset aspek berlabel untuk Yelp). Kontribusi novel proyek ini ada di
mekanisme Attention-Gated Fusion (Fase 2), bukan di teknik ekstraksi aspek
ini.

TIGA cara mengonsumsi skor per-aspek, masing-masing method independen
(tidak berbagi kode -- zero risiko regresi antar varian):
1. `score_dataframe()`: rata-rata polos antar aspek yang match -> 1 skalar
   (mode "mean" di run_baseline_absa.py). RMSE empiris jauh lebih buruk
   dari SA global -- rata-rata polos membuang banyak informasi.
2. `score_dataframe_per_aspect()`: TANPA agregasi -- vektor mentah per
   aspek dikirim langsung ke fusion (mode "concat"). RMSE empiris setara
   SA global (tidak signifikan berbeda, lihat run_significance_test.py).
3. `score_dataframe_confidence_weighted()`: rata-rata BERBOBOT confidence
   antar aspek yang match -> 1 skalar (mode "confidence_mean"). Menguji
   hipotesis dari paper IEEE penulis sebelumnya: apakah confidence-aware
   weighting bisa memperbaiki kegagalan rata-rata polos di (1).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_ASPECT_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "restaurant": {
        "food": ["food", "dish", "meal", "taste", "flavor", "flavour", "menu", "delicious", "portion"],
        "service": ["service", "staff", "waiter", "waitress", "server", "friendly", "rude", "slow service"],
        "price": ["price", "expensive", "cheap", "affordable", "overpriced", "value", "worth"],
        "ambiance": ["ambiance", "ambience", "atmosphere", "decor", "music", "noisy", "cozy", "clean"],
    },
    "hotel": {
        "room": ["room", "bed", "bathroom", "shower", "clean room", "spacious", "view"],
        "service": ["service", "staff", "receptionist", "front desk", "friendly", "helpful", "rude"],
        "location": ["location", "close to", "nearby", "walking distance", "downtown", "far from"],
        "price": ["price", "expensive", "cheap", "affordable", "overpriced", "value", "worth"],
        "cleanliness": ["clean", "dirty", "spotless", "smell", "hygiene", "housekeeping"],
    },
}


def _ensure_sentence_tokenizer() -> None:
    """Download resource NLTK yang dibutuhkan `sent_tokenize()`.

    SENGAJA tidak menambah ke `preprocessing.ensure_nltk_resources()` (yang
    sudah tervalidasi/dipakai stage lain) -- nltk versi terbaru (>=3.8.2)
    butuh resource "punkt_tab" (bukan cuma "punkt") untuk sent_tokenize,
    yang belum di-download di sana. Modul ABSA baru ini yang menanggung
    risiko resource tambahan, bukan kode preprocessing yang sudah stabil.
    """
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info("Mengunduh resource NLTK: %s", resource)
            nltk.download(resource, quiet=True)


@dataclass
class ABSAConfig:
    domain: str = "restaurant"
    aspect_keywords: dict[str, list[str]] | None = None  # None -> DEFAULT_ASPECT_KEYWORDS[domain]
    aggregation: str = "mean"  # rata-rata skor antar aspek yang match (satu-satunya yg didukung)
    min_sentence_chars: int = 3

    def resolved_keywords(self) -> dict[str, list[str]]:
        if self.aspect_keywords is not None:
            return self.aspect_keywords
        if self.domain not in DEFAULT_ASPECT_KEYWORDS:
            raise ValueError(
                f"Tidak ada DEFAULT_ASPECT_KEYWORDS untuk domain '{self.domain}' -- "
                "sediakan aspect_keywords secara eksplisit di ABSAConfig."
            )
        return DEFAULT_ASPECT_KEYWORDS[self.domain]


class KeywordAspectSentimentScorer:
    """Skor sentimen per-aspek berbasis keyword matching + model SA yang
    SUDAH di-checkpoint (tidak ada training baru di kelas ini)."""

    def __init__(self, sentiment_model, config: ABSAConfig | None = None):
        self.sentiment_model = sentiment_model
        self.config = config or ABSAConfig()
        self.aspect_keywords = {
            aspect: [kw.lower() for kw in kws]
            for aspect, kws in self.config.resolved_keywords().items()
        }
        if self.config.aggregation != "mean":
            raise ValueError(
                f"aggregation '{self.config.aggregation}' belum didukung -- cuma 'mean'."
            )
        _ensure_sentence_tokenizer()

    def _split_sentences(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        sentences = nltk.sent_tokenize(text)
        return [s for s in sentences if len(s.strip()) >= self.config.min_sentence_chars]

    def _match_aspects(self, sentences: list[str]) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = defaultdict(list)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for aspect, keywords in self.aspect_keywords.items():
                if any(kw in sentence_lower for kw in keywords):
                    matches[aspect].append(sentence)
        return dict(matches)

    def score_dataframe(self, df: pd.DataFrame, text_column: str = "text_bert") -> np.ndarray:
        """Drop-in pengganti GlobalSentimentBERT.predict_proba(): satu float
        [0,1] per baris df, urutan sama.

        Algoritma (lihat docstring modul untuk alasan desain):
        1. Sentence-split + match aspek per baris.
        2. Baris tanpa aspek match sama sekali -> fallback skor SELURUH teks
           review (bukan NaN) -- selalu ada 1 skor per baris.
        3. Semua teks (baik hasil gabungan kalimat per-aspek maupun fallback
           whole-review) di-flatten jadi SATU list -> SATU panggilan
           predict_proba() (bukan loop per-baris), supaya efisien.
        4. Regroup skor per baris -> rata-rata antar aspek yang match ->
           1 skalar per baris.
        """
        texts = df[text_column].fillna("").tolist()

        flat_texts: list[str] = []
        flat_row_idx: list[int] = []

        for row_idx, text in enumerate(texts):
            sentences = self._split_sentences(text)
            aspect_matches = self._match_aspects(sentences)
            if not aspect_matches:
                flat_texts.append(text)
                flat_row_idx.append(row_idx)
            else:
                for _aspect, matched_sentences in aspect_matches.items():
                    flat_texts.append(" ".join(matched_sentences))
                    flat_row_idx.append(row_idx)

        avg_calls_per_row = len(flat_texts) / len(texts) if texts else 0.0
        logger.info(
            "ABSA: %d baris -> %d panggilan teks ke model SA (rata-rata %.2fx lipat "
            "dibanding skor SA global, karena tiap aspek yang match discor terpisah). "
            "Estimasi waktu proporsional dengan kelipatan ini -- lihat progress bar di bawah.",
            len(texts),
            len(flat_texts),
            avg_calls_per_row,
        )
        flat_scores = self.sentiment_model.predict_proba(flat_texts)

        per_row_scores: dict[int, list[float]] = defaultdict(list)
        for row_idx, score in zip(flat_row_idx, flat_scores):
            per_row_scores[row_idx].append(float(score))

        final_scores = np.empty(len(texts), dtype=np.float32)
        for row_idx in range(len(texts)):
            values = per_row_scores.get(row_idx, [0.5])
            final_scores[row_idx] = float(np.mean(values))
        return final_scores

    def score_dataframe_per_aspect(self, df: pd.DataFrame, text_column: str = "text_bert") -> pd.DataFrame:
        """Skor sentimen per-aspek TANPA agregasi -- 1 kolom per aspek (nama
        kolom = nama aspek), untuk varian ablasi "ABSA-concat": fusion
        menerima vektor skor mentah, bukan drop-in sentiment_score tunggal
        seperti score_dataframe(). Method INDEPENDEN dari score_dataframe()
        (tidak berbagi kode) supaya tidak ada risiko regresi ke hasil
        score_dataframe() yang sudah divalidasi.

        Baris yang tidak menyebut aspek tertentu diisi 1 skor fallback
        SELURUH teks review untuk aspek itu (bukan NaN) -- fallback dihitung
        SEKALI per baris (bukan berulang per aspek yang hilang) lalu dipakai
        ulang utk semua aspek yang hilang di baris itu, supaya jumlah
        panggilan ke model SA tetap efisien (mirip skala dengan
        score_dataframe(), bukan N_aspek x lipat).
        """
        texts = df[text_column].fillna("").tolist()
        aspect_names = list(self.aspect_keywords.keys())

        flat_texts: list[str] = []
        flat_row_idx: list[int] = []
        flat_aspect: list[str | None] = []  # None = skor fallback whole-review baris ini

        for row_idx, text in enumerate(texts):
            sentences = self._split_sentences(text)
            aspect_matches = self._match_aspects(sentences)
            for aspect, matched_sentences in aspect_matches.items():
                flat_texts.append(" ".join(matched_sentences))
                flat_row_idx.append(row_idx)
                flat_aspect.append(aspect)
            if any(aspect not in aspect_matches for aspect in aspect_names):
                flat_texts.append(text)
                flat_row_idx.append(row_idx)
                flat_aspect.append(None)

        avg_calls_per_row = len(flat_texts) / len(texts) if texts else 0.0
        logger.info(
            "ABSA-concat: %d baris -> %d panggilan teks ke model SA (rata-rata %.2fx lipat). "
            "Umumnya lebih tinggi dari mode mean karena hampir tiap baris juga butuh 1 "
            "skor fallback untuk aspek yang tidak disebut.",
            len(texts),
            len(flat_texts),
            avg_calls_per_row,
        )
        flat_scores = self.sentiment_model.predict_proba(flat_texts)

        row_aspect_scores: dict[int, dict[str, float]] = defaultdict(dict)
        row_fallback: dict[int, float] = {}
        for row_idx, aspect, score in zip(flat_row_idx, flat_aspect, flat_scores):
            if aspect is None:
                row_fallback[row_idx] = float(score)
            else:
                row_aspect_scores[row_idx][aspect] = float(score)

        result_array = np.empty((len(texts), len(aspect_names)), dtype=np.float32)
        for row_idx in range(len(texts)):
            fallback = row_fallback.get(row_idx, 0.5)
            for col_idx, aspect in enumerate(aspect_names):
                result_array[row_idx, col_idx] = row_aspect_scores[row_idx].get(aspect, fallback)
        return pd.DataFrame(result_array, columns=aspect_names)

    @staticmethod
    def _sentiment_confidence(score: float) -> float:
        """Margin skor dari 0,5 sbg proxy confidence prediksi sentimen --
        skor dekat 0,5 (ambigu) = confidence rendah, skor dekat 0/1 (jelas
        positif/negatif) = confidence tinggi. TIDAK butuh panggilan model
        tambahan -- turunan murni dari skor predict_proba() yang sudah ada.
        """
        return abs(score - 0.5) * 2.0

    @staticmethod
    def _evidence_confidence(n_sentences: int, cap: int = 3) -> float:
        """Proxy confidence dari jumlah kalimat yang match utk aspek ini --
        lebih banyak bukti tekstual = lebih yakin. Cap di `cap` kalimat (di
        atas itu dianggap sudah cukup bukti, tidak naik linear tanpa batas).
        """
        return min(n_sentences / cap, 1.0)

    def score_dataframe_confidence_weighted(
        self, df: pd.DataFrame, text_column: str = "text_bert", min_confidence: float = 0.05
    ) -> np.ndarray:
        """Drop-in pengganti GlobalSentimentBERT.predict_proba() SEPERTI
        score_dataframe(), TAPI agregasi antar-aspek pakai RATA-RATA
        BERBOBOT confidence (bukan rata-rata polos) -- method INDEPENDEN
        (tidak berbagi kode dengan score_dataframe(), zero risiko regresi
        ke hasil ABSA-mean yang sudah divalidasi).

        Confidence per (baris, aspek) = rata-rata dari:
        1. sentiment_confidence: margin skor dari 0,5 (skor yang SUDAH
           dihitung -- TIDAK ada panggilan BERT tambahan sama sekali
           dibanding score_dataframe(), cuma bookkeeping ekstra).
        2. evidence_confidence: seberapa banyak kalimat yang match untuk
           aspek itu (lebih banyak bukti tekstual = lebih yakin).
        Baris fallback (whole-review, 0 aspek match) otomatis dapat
        evidence_confidence=0 -- konsisten dengan intuisi bahwa estimasi
        fallback kurang spesifik/kurang bisa diandalkan.

        `min_confidence`: batas bawah supaya tidak ada pembagian dengan
        total bobot nol -- skor tetap ikut kontribusi minimal, bukan
        diabaikan total.
        """
        texts = df[text_column].fillna("").tolist()

        flat_texts: list[str] = []
        flat_row_idx: list[int] = []
        flat_n_sentences: list[int] = []

        for row_idx, text in enumerate(texts):
            sentences = self._split_sentences(text)
            aspect_matches = self._match_aspects(sentences)
            if not aspect_matches:
                flat_texts.append(text)
                flat_row_idx.append(row_idx)
                flat_n_sentences.append(0)  # fallback -- tidak ada bukti aspek spesifik
            else:
                for _aspect, matched_sentences in aspect_matches.items():
                    flat_texts.append(" ".join(matched_sentences))
                    flat_row_idx.append(row_idx)
                    flat_n_sentences.append(len(matched_sentences))

        logger.info(
            "ABSA-confidence: %d baris -> %d panggilan teks ke model SA (sama seperti mode "
            "mean -- confidence dihitung dari skor & jumlah kalimat yang SUDAH ada, tanpa "
            "panggilan tambahan).",
            len(texts), len(flat_texts),
        )
        flat_scores = self.sentiment_model.predict_proba(flat_texts)

        per_row_weighted: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row_idx, score, n_sentences in zip(flat_row_idx, flat_scores, flat_n_sentences):
            sentiment_conf = self._sentiment_confidence(float(score))
            evidence_conf = self._evidence_confidence(n_sentences)
            confidence = max((sentiment_conf + evidence_conf) / 2.0, min_confidence)
            per_row_weighted[row_idx].append((float(score), confidence))

        final_scores = np.empty(len(texts), dtype=np.float32)
        for row_idx in range(len(texts)):
            pairs = per_row_weighted.get(row_idx, [(0.5, min_confidence)])
            scores_arr = np.array([p[0] for p in pairs])
            weights_arr = np.array([p[1] for p in pairs])
            final_scores[row_idx] = float(np.sum(scores_arr * weights_arr) / np.sum(weights_arr))
        return final_scores

    def compute_aspect_evidence_counts(self, df: pd.DataFrame, text_column: str = "text_bert") -> pd.DataFrame:
        """Hitung jumlah kalimat yang match per aspek, TANPA panggilan
        predict_proba() sama sekali (CPU-only, murah -- cuma sentence-split
        + keyword matching). Dipakai bareng skor aspek yang SUDAH ada (dari
        score_dataframe_per_aspect() atau cache absa_aspect_scores.csv) utk
        menghitung confidence tanpa perlu inference BERT ulang -- lihat
        mode "concat_confidence" di run_baseline_absa.py.
        """
        texts = df[text_column].fillna("").tolist()
        aspect_names = list(self.aspect_keywords.keys())
        counts = np.zeros((len(texts), len(aspect_names)), dtype=np.int32)

        for row_idx, text in enumerate(texts):
            sentences = self._split_sentences(text)
            aspect_matches = self._match_aspects(sentences)
            for col_idx, aspect in enumerate(aspect_names):
                if aspect in aspect_matches:
                    counts[row_idx, col_idx] = len(aspect_matches[aspect])
        return pd.DataFrame(counts, columns=aspect_names)

    def aspect_coverage_report(self, df: pd.DataFrame, text_column: str = "text_bert") -> dict:
        """Diagnostik cakupan aspek (WAJIB dilog/disimpan tiap run nyata,
        bukan cuma nice-to-have): cakupan rendah = ablasi ABSA kurang
        bermakna karena banyak baris jatuh ke fallback whole-review lagi."""
        texts = df[text_column].fillna("").tolist()
        total = len(texts)
        aspect_counts = {aspect: 0 for aspect in self.aspect_keywords}
        n_with_any_match = 0

        for text in texts:
            sentences = self._split_sentences(text)
            matches = self._match_aspects(sentences)
            if matches:
                n_with_any_match += 1
            for aspect in matches:
                aspect_counts[aspect] += 1

        return {
            "n_reviews": total,
            "pct_with_any_aspect_match": n_with_any_match / total if total else 0.0,
            "aspect_match_counts": aspect_counts,
            "aspect_match_pct": {
                aspect: (count / total if total else 0.0) for aspect, count in aspect_counts.items()
            },
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton ABSA keyword-based -- jalankan via run_baseline_absa.py "
        "dengan GlobalSentimentBERT yang sudah di-checkpoint, jangan berdiri sendiri."
    )
