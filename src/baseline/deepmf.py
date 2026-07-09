"""
src/baseline/deepmf.py

Reimplementasi Deep Matrix Factorization untuk collaborative filtering,
mengikuti spesifikasi arsitektur pada proposal (Embedding 128, Deep layers
[256,128,64,32], dropout 0.3) yang dipakai identik oleh baseline maupun
sebagai salah satu stream A2-FusionRS -- perbedaan utama ada di fusion
layer, bukan di modul DeepMF ini sendiri.

Catatan: hyperparameter batch_size=512, lr=0.001, negative_sampling 1:4
sesuai proposal bagian E (Metode).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class DeepMFConfig:
    embedding_dim: int = 128
    hidden_layers: tuple[int, ...] = (256, 128, 64, 32)
    dropout: float = 0.3
    batch_size: int = 512
    learning_rate: float = 0.001
    epochs: int = 20
    negative_sampling_ratio: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class InteractionDataset(Dataset):
    """Dataset user-item dengan negative sampling.

    Rating asli (1-5) dinormalisasi ke (0,1) untuk konsisten dengan
    prediction head sigmoid; denormalisasi dilakukan di evaluasi (metrics.py)
    sebelum menghitung RMSE/MAE pada skala asli 1-5, agar hasil tetap
    comparable dengan paper lain yang melaporkan RMSE pada skala rating asli.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        n_items: int,
        negative_ratio: int = 4,
        rating_scale: tuple[float, float] = (1.0, 5.0),
        seed: int = 42,
    ):
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.n_items = n_items
        self.rating_min, self.rating_max = rating_scale

        users = df["user_id"].map(user2idx).values
        items = df["business_id"].map(item2idx).values
        ratings = df["stars"].values.astype(np.float32)
        ratings_norm = (ratings - self.rating_min) / (self.rating_max - self.rating_min)

        pos_users, pos_items, pos_labels = users, items, ratings_norm

        # Negative sampling: item yang TIDAK pernah berinteraksi dengan user tsb
        user_positive_items: dict[int, set] = {}
        for u, i in zip(users, items):
            user_positive_items.setdefault(u, set()).add(i)

        rng = np.random.default_rng(seed)
        neg_users, neg_items = [], []
        for u in users:
            seen = user_positive_items.get(u, set())
            for _ in range(negative_ratio):
                neg_item = rng.integers(0, n_items)
                attempts = 0
                while neg_item in seen and attempts < 10:
                    neg_item = rng.integers(0, n_items)
                    attempts += 1
                neg_users.append(u)
                neg_items.append(neg_item)

        neg_labels = np.zeros(len(neg_users), dtype=np.float32)

        self.users = np.concatenate([pos_users, np.array(neg_users)])
        self.items = np.concatenate([pos_items, np.array(neg_items)])
        self.labels = np.concatenate([pos_labels, neg_labels])

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "item": torch.tensor(self.items[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class DeepMFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, config: DeepMFConfig):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, config.embedding_dim)

        layers = []
        input_dim = config.embedding_dim
        for hidden_dim in config.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        self.deep_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, return_latent: bool = False):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        interaction = u_emb * i_emb  # element-wise product
        latent = self.deep_layers(interaction)
        pred = torch.sigmoid(self.output_layer(latent)).squeeze(-1)

        if return_latent:
            return pred, latent
        return pred


class DeepMFTrainer:
    def __init__(self, n_users: int, n_items: int, config: DeepMFConfig | None = None):
        self.config = config or DeepMFConfig()
        self.model = DeepMFModel(n_users, n_items, self.config).to(self.config.device)

    def fit(self, train_dataset: InteractionDataset, val_dataset: InteractionDataset | None = None) -> None:
        loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Model DeepMF ini cenderung overfit setelah beberapa epoch awal
        # (val RMSE memburuk terus-menerus di observasi run penuh), sementara
        # config.epochs tetap dijalankan penuh tanpa early stopping. Untuk
        # itu kita lacak state_dict dengan val RMSE terbaik dan restore di
        # akhir, supaya model yang dipakai stream selanjutnya BUKAN otomatis
        # model epoch terakhir (yang bisa jadi justru yang terburuk).
        best_val_rmse = float("inf")
        best_state_dict = None

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                user = batch["user"].to(self.config.device)
                item = batch["item"].to(self.config.device)
                label = batch["label"].to(self.config.device)

                pred = self.model(user, item)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            log_msg = f"Epoch {epoch + 1}/{self.config.epochs} - train MSE: {avg_loss:.4f}"
            if val_dataset is not None:
                val_rmse = self.evaluate_rmse(val_dataset)
                log_msg += f" - val RMSE (normalized): {val_rmse:.4f}"
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    log_msg += " (terbaik sejauh ini, disimpan)"
            logger.info(log_msg)

        if val_dataset is not None and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(
                "Restore bobot model DeepMF dari epoch dengan val RMSE terbaik "
                "(%.4f) -- bukan otomatis model epoch terakhir.",
                best_val_rmse,
            )

    @torch.no_grad()
    def evaluate_rmse(self, dataset: InteractionDataset) -> float:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        squared_errors = []
        for batch in loader:
            user = batch["user"].to(self.config.device)
            item = batch["item"].to(self.config.device)
            label = batch["label"].to(self.config.device)
            pred = self.model(user, item)
            squared_errors.append(((pred - label) ** 2).cpu().numpy())
        self.model.train()
        return float(np.sqrt(np.concatenate(squared_errors).mean()))

    @torch.no_grad()
    def predict(
        self,
        df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        rating_scale: tuple[float, float] = (1.0, 5.0),
        unseen_fallback: str = "global_mean",
    ) -> np.ndarray:
        """Prediksi rating pada SKALA ASLI (mis. 1-5) untuk baris user-item
        pada df tertentu (train/val/test) -- dipakai sebagai salah satu
        stream input ke fusion layer (bukan untuk ranking, murni regresi).

        Parameters
        ----------
        unseen_fallback : str
            Strategi untuk user/item yang tidak dikenal (tidak ada di
            user2idx/item2idx -- misal cold-start di test set):
            "global_mean" -> pakai rata-rata rating skala (rating_min+rating_max)/2
            "error" -> raise exception (dipakai untuk debugging ketat)
        """
        self.model.eval()
        rating_min, rating_max = rating_scale
        scale_range = rating_max - rating_min

        user_idx = df["user_id"].map(user2idx)
        item_idx = df["business_id"].map(item2idx)
        unknown_mask = user_idx.isna() | item_idx.isna()

        if unknown_mask.any():
            n_unknown = int(unknown_mask.sum())
            if unseen_fallback == "error":
                raise KeyError(
                    f"{n_unknown} baris memiliki user/item yang tidak dikenal "
                    "model (kemungkinan cold-start) -- set unseen_fallback="
                    "'global_mean' jika ingin fallback otomatis."
                )
            logger.warning(
                "%d baris (dari %d) memiliki user/item unknown (cold-start), "
                "diisi dengan rating rata-rata skala (%.2f) sebagai fallback.",
                n_unknown,
                len(df),
                rating_min + scale_range / 2,
            )

        preds = np.full(len(df), rating_min + scale_range / 2, dtype=np.float32)
        known_rows = ~unknown_mask.values

        if known_rows.any():
            u_tensor = torch.tensor(
                user_idx[known_rows].astype(int).values, dtype=torch.long, device=self.config.device
            )
            i_tensor = torch.tensor(
                item_idx[known_rows].astype(int).values, dtype=torch.long, device=self.config.device
            )
            batch_size = self.config.batch_size
            known_preds = []
            for start in range(0, len(u_tensor), batch_size):
                end = start + batch_size
                p = self.model(u_tensor[start:end], i_tensor[start:end])
                known_preds.append(p.cpu().numpy())
            known_preds = np.concatenate(known_preds)
            # denormalisasi dari (0,1) sigmoid ke skala rating asli
            preds[known_rows] = known_preds * scale_range + rating_min

        self.model.train()
        return preds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton DeepMF -- jalankan via run_baseline.py setelah split & "
        "user2idx/item2idx mapping tersedia."
    )
