"""
src/baseline/sentiment_bert.py

Reimplementasi komponen Sentiment Analysis (SA) dari Darraz et al. (2025):
BERT-base-uncased fine-tuned untuk SA GLOBAL (bukan per-aspek -- ini yang
disebut proposal sebagai limitasi baseline yang diatasi A2-FusionRS lewat
ABSA). Modul ini SENGAJA dibuat sederhana/global agar jadi pembanding yang
fair terhadap stream ABSA-BERT di A2-FusionRS nanti.

Hyperparameter mengikuti Table 1 baseline paper: AdamW, learning rate 1e-5.
(Proposal A2-FusionRS memakai learning rate 2e-5 untuk ABSA-BERT -- ini
perbedaan yang disengaja dari desain masing-masing, bukan salah ketik;
jangan disamakan saat reimplementasi baseline ini.)

PERINGATAN PRODUCTION-READINESS:
- Belum ada label sentiment ground-truth eksplisit di dataset Yelp (yang ada
  hanya `stars`). Baseline paper kemungkinan men-derive label sentiment dari
  rating (misal >=4 = positif, <=2 = negatif, 3 = netral/dibuang) -- perlu
  dikonfirmasi ulang terhadap detail metodologi paper sebelum training,
  karena strategi labeling ini SANGAT menentukan validitas evaluasi (lih.
  diskusi RMSE anomali sebelumnya: jika label sentiment diturunkan langsung
  dari rating yang sama dengan target prediksi RMSE, ini berpotensi jadi
  sumber data leakage).
- Kode ini belum diuji end-to-end pada dataset riil (menunggu file dataset
  diunduh manual oleh pengguna).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def derive_sentiment_label(stars: float) -> int:
    """Turunkan label sentiment biner dari rating bintang.

    ASUMSI YANG PERLU DIVALIDASI: baseline paper melaporkan akurasi SA
    90%/87%, mengindikasikan task klasifikasi (kemungkinan biner atau
    3-kelas), bukan regresi langsung. Skema di bawah adalah skema paling
    umum di literatur SA-on-ratings, dipakai sebagai starting point --
    WAJIB dicek ulang terhadap teks lengkap paper (bagian metodologi SA)
    sebelum dipakai sebagai ground truth final.
    """
    if stars >= 4:
        return 1  # positif
    elif stars <= 2:
        return 0  # negatif
    else:
        raise ValueError(
            f"stars={stars} berada di zona netral (3) -- baseline kemungkinan "
            "membuang kelas netral ini dari training SA. Filter baris ini "
            "sebelum memanggil fungsi ini, jangan biarkan exception ini terjadi "
            "di tengah loop training."
        )


class ReviewSentimentDataset(Dataset):
    """Dataset dengan PRE-TOKENISASI (batch, sekali di __init__), bukan
    tokenisasi per-sampel di __getitem__. Perubahan ini penting untuk
    kecepatan di GPU: tanpa pre-tokenisasi, CPU jadi bottleneck karena
    tokenizer dipanggil satu-per-satu setiap kali DataLoader mengambil
    sampel, sementara GPU menunggu idle -- terutama terasa di Colab
    dengan GPU cepat (T4/L4/A100) tapi CPU host yang biasa-biasa saja.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


@dataclass
class SentimentBertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 1e-5
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # --- Opsi khusus percepatan di GPU (Colab) ---
    num_workers: int = 2          # parallel data loading; set 0 jika di Windows lokal tanpa GPU
    pin_memory: bool = torch.cuda.is_available()
    use_amp: bool = torch.cuda.is_available()  # mixed precision (fp16) -- speedup ~1.5-2x di T4/L4/A100


class GlobalSentimentBERT:
    """Wrapper training/inference untuk baseline SA global."""

    def __init__(self, config: SentimentBertConfig | None = None):
        self.config = config or SentimentBertConfig()
        logger.info(
            "Menggunakan device: %s (AMP=%s, num_workers=%d)",
            self.config.device,
            self.config.use_amp,
            self.config.num_workers,
        )
        if self.config.device == "cuda":
            # cudnn.benchmark mempercepat training dengan mencari algoritma
            # konvolusi/matmul tercepat untuk ukuran input yang konsisten
            # (semua batch di sini punya max_length yang sama karena padding
            # "max_length", jadi aman diaktifkan -- tidak ada overhead
            # re-benchmark berulang akibat ukuran input yang berubah-ubah).
            torch.backends.cudnn.benchmark = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, num_labels=2
        ).to(self.config.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        train_texts = train_df["text_bert"].tolist()
        train_labels = train_df["sentiment_label"].tolist()

        train_dataset = ReviewSentimentDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        logger.info(
            "Mulai training: %d baris, %d batch/epoch, %d epoch (device=%s, AMP=%s). "
            "Progress per-batch akan tampil di bawah -- jika device='cpu' dan "
            "tidak ada progress bar bergerak sama sekali dalam beberapa menit, "
            "kemungkinan proses benar-benar hang, bukan sekadar lambat.",
            len(train_texts),
            len(train_loader),
            self.config.epochs,
            self.config.device,
            self.config.use_amp,
        )

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                unit="batch",
            )
            for batch in progress_bar:
                optimizer.zero_grad(set_to_none=True)
                input_ids = batch["input_ids"].to(self.config.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=True)
                labels = batch["label"].to(self.config.device, non_blocking=True)

                # Mixed precision (autocast) -- forward pass dihitung dalam
                # fp16 (di GPU Tensor Core seperti T4/L4/A100) untuk speedup,
                # GradScaler menangani skala gradien agar tidak underflow.
                # Otomatis no-op (setara precision biasa) kalau use_amp=False
                # atau device='cpu'.
                with torch.autocast(device_type="cuda", enabled=self.config.use_amp):
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            logger.info("Epoch %d/%d - avg train loss: %.4f", epoch + 1, self.config.epochs, avg_loss)

            if val_df is not None:
                val_acc = self.evaluate(val_df)
                logger.info("Epoch %d - val accuracy: %.4f", epoch + 1, val_acc)

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probabilitas kelas positif -- dipakai sebagai skor sentimen
        global yang masuk ke fusion layer (bukan hanya label biner)."""
        self.model.eval()
        dataset = ReviewSentimentDataset(
            texts, [0] * len(texts), self.tokenizer, self.config.max_length
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        probs = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=self.config.use_amp):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1]
            probs.append(batch_probs.cpu().numpy())
        return np.concatenate(probs)

    @torch.no_grad()
    def evaluate(self, df: pd.DataFrame) -> float:
        texts = df["text_bert"].tolist()
        labels = np.array(df["sentiment_label"].tolist())
        probs = self.predict_proba(texts)
        preds = (probs >= 0.5).astype(int)
        return float((preds == labels).mean())

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model SA disimpan ke %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Modul ini adalah skeleton -- jalankan lewat run_baseline.py setelah "
        "data_loader dan split_generator dieksekusi, jangan dijalankan berdiri "
        "sendiri tanpa data yang sudah displit."
    )
