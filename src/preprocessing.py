"""
src/preprocessing.py

Preprocessing teks review sebelum masuk ke tahap sentiment analysis.
Dipakai bersama baik oleh pipeline baseline maupun A2-FusionRS -- pastikan
tidak ada perbedaan preprocessing antar model yang dibandingkan, karena
perbedaan tokenisasi/cleaning bisa jadi confounding factor pada hasil akhir.
"""

from __future__ import annotations

import logging
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


def ensure_nltk_resources() -> None:
    """Unduh resource NLTK yang diperlukan jika belum ada di environment.

    Dipisah jadi fungsi agar bisa dipanggil eksplisit sekali di awal notebook/
    script, daripada silent-download di tengah proses batch besar.
    """
    resources = ["stopwords", "wordnet", "omw-1.4", "punkt"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            logger.info("Mengunduh resource NLTK: %s", res)
            nltk.download(res, quiet=True)


class TextPreprocessor:
    """Preprocessing teks review: cleaning, stopword removal, lemmatization.

    Catatan desain: untuk input ke BERT, sebagian besar cleaning agresif
    (stopword removal, lemmatization) SEBAIKNYA TIDAK diterapkan sebelum
    tokenisasi BERT, karena BERT dilatih pada teks natural dan subword
    tokenizer-nya sudah menangani variasi kata. Kelas ini menyediakan dua
    mode: `clean_for_bert()` (minimal cleaning) dan `clean_for_tfidf()`
    (full cleaning, dipakai untuk representasi TF-IDF di stream CBF).
    """

    def __init__(self, language: str = "english"):
        ensure_nltk_resources()
        self.stopwords = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_for_bert(self, text: str) -> str:
        """Minimal cleaning: hapus whitespace berlebih & karakter kontrol,
        pertahankan casing dan struktur kalimat asli untuk tokenizer BERT."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\x20-\x7E]", "", text)  # buang karakter non-ASCII aneh
        return text

    def clean_for_tfidf(self, text: str) -> str:
        """Full cleaning untuk representasi TF-IDF (stream Content-Based):
        lowercase, buang stopword, lemmatisasi."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        df = df.copy()
        df["text_bert"] = df[text_column].apply(self.clean_for_bert)
        df["text_tfidf"] = df[text_column].apply(self.clean_for_tfidf)

        empty_after_clean = (df["text_bert"].str.len() == 0).sum()
        if empty_after_clean > 0:
            logger.warning(
                "%d baris menjadi teks kosong setelah cleaning -- akan "
                "menyebabkan masalah di tahap sentiment analysis, "
                "pertimbangkan untuk membuang baris ini.",
                empty_after_clean,
            )
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = pd.DataFrame(
        {"text": ["Good gyros, clean and friendly staff!!!  \n\n", "Terrible service..."]}
    )
    pre = TextPreprocessor()
    print(pre.preprocess_dataframe(sample))
