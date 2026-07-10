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
dataset aspek berlabel untuk Yelp) dan TANPA confidence-weighting (skor
sentimen per aspek saja). Kontribusi novel proyek ini ada di mekanisme
Attention-Gated Fusion (Fase 2), bukan di teknik ekstraksi aspek ini.

PENTING: skor per aspek di sini di-AGREGASI jadi SATU skalar per baris
(rata-rata antar aspek yang match) sebelum dipakai run_baseline_absa.py --
supaya jadi drop-in replacement persis untuk kolom `sentiment_score` yang
sudah ada, TANPA perlu mengubah cbf_clustering.py/fusion_nmf_dt.py sama
sekali. Mengekspos vektor skor per-aspek penuh ke fusion (bukan cuma rata-
rata) adalah pekerjaan Fase 2 (Attention-Gated Fusion yang sesungguhnya).
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

        flat_scores = self.sentiment_model.predict_proba(flat_texts)

        per_row_scores: dict[int, list[float]] = defaultdict(list)
        for row_idx, score in zip(flat_row_idx, flat_scores):
            per_row_scores[row_idx].append(float(score))

        final_scores = np.empty(len(texts), dtype=np.float32)
        for row_idx in range(len(texts)):
            values = per_row_scores.get(row_idx, [0.5])
            final_scores[row_idx] = float(np.mean(values))
        return final_scores

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
