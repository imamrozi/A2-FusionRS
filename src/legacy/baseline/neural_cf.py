"""
src/baseline/neural_cf.py

Baseline collaborative-filtering NEURAL EKSTERNAL (NeuMF, DeepFM) sebagai
pembanding state-of-the-art terhadap A2-FusionRS -- pembanding eksternal yang
diminta reviewer Q1 (di luar baseline internal A2-IRM/Darraz et al.). MURNI
CF dari sinyal rating (user_id, business_id, stars), TANPA sentiment/content,
TIDAK terintegrasi ke pipeline fusion. Dijalankan & dievaluasi berdiri
sendiri lewat run_neural_cf.py memakai split IDENTIK dengan semua model lain.

Dua arsitektur kanonik direimplementasi setia pada paper aslinya, TAPI
diadaptasi ke prediksi rating eksplisit (regresi RMSE), bukan ranking
implicit-feedback aslinya -- supaya sebanding pada metrik RMSE/MAE yang jadi
fokus penelitian ini:

- NeuMF (He et al., 2017, WWW, "Neural Collaborative Filtering"): gabungan
  GMF (generalized matrix factorization, hasil kali elemen embedding) +
  MLP (embedding terpisah, dikonkat lalu MLP), lapisan NeuMF terakhir
  mengonkat kedua jalur. Output sigmoid pada rating ternormalisasi (0,1)
  -- konvensi SAMA dengan DeepMFModel (deepmf.py) supaya semua model neural
  di repo ini setara.
- DeepFM (Guo et al., 2017, IJCAI, "DeepFM: A Factorization-Machine based
  Neural Network"): komponen FM (orde-1 linear + orde-2 interaksi
  faktorisasi) paralel dengan komponen deep (MLP atas embedding
  terkonkat), berbagi embedding yang sama. Dengan 2 field (user, item),
  suku orde-2 FM tereduksi menjadi <v_user, v_item>.

Konsisten dengan deepmf.py: rating dinormalisasi ke (0,1) saat training,
didenormalisasi ke skala asli 1-5 saat evaluasi; cold-start user/item baru
di test set fallback ke rating rata-rata skala; bobot epoch dengan val RMSE
terbaik di-restore di akhir (bukan otomatis epoch terakhir). Regresi pada
rating teramati saja, TANPA negative sampling (sesuai temuan empiris di
deepmf.py bahwa negative sampling justru overfit untuk prediksi rating).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class NeuralCFConfig:
    model: str = "neumf"  # "neumf" | "deepfm"
    embedding_dim: int = 64
    mlp_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-6
    epochs: int = 20
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RatingDataset(Dataset):
    """Dataset rating teramati (regresi eksplisit, TANPA negative sampling).
    Rating 1-5 dinormalisasi ke (0,1) agar konsisten dengan output sigmoid."""

    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: dict,
        item2idx: dict,
        rating_scale: tuple[float, float] = (1.0, 5.0),
    ):
        rating_min, rating_max = rating_scale
        self.users = df["user_id"].map(user2idx).values.astype(np.int64)
        self.items = df["business_id"].map(item2idx).values.astype(np.int64)
        ratings = df["stars"].values.astype(np.float32)
        self.labels = (ratings - rating_min) / (rating_max - rating_min)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "item": torch.tensor(self.items[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class NeuMFModel(nn.Module):
    """NeuMF (He et al., 2017) diadaptasi ke regresi rating (output sigmoid)."""

    def __init__(self, n_users: int, n_items: int, config: NeuralCFConfig):
        super().__init__()
        d = config.embedding_dim
        # Embedding TERPISAH untuk jalur GMF dan MLP (sesuai paper).
        self.user_emb_gmf = nn.Embedding(n_users, d)
        self.item_emb_gmf = nn.Embedding(n_items, d)
        self.user_emb_mlp = nn.Embedding(n_users, d)
        self.item_emb_mlp = nn.Embedding(n_items, d)

        layers = []
        input_dim = 2 * d
        for hidden_dim in config.mlp_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Lapisan NeuMF: konkat vektor GMF (d) + keluaran MLP (mlp_layers[-1]).
        self.output_layer = nn.Linear(d + config.mlp_layers[-1], 1)

        for emb in (self.user_emb_gmf, self.item_emb_gmf, self.user_emb_mlp, self.item_emb_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        gmf = self.user_emb_gmf(user_idx) * self.item_emb_gmf(item_idx)
        mlp_in = torch.cat([self.user_emb_mlp(user_idx), self.item_emb_mlp(item_idx)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        concat = torch.cat([gmf, mlp_out], dim=-1)
        return torch.sigmoid(self.output_layer(concat)).squeeze(-1)


class DeepFMModel(nn.Module):
    """DeepFM (Guo et al., 2017) untuk 2 field (user, item), regresi rating.

    Suku orde-2 FM untuk 2 field = <v_user, v_item>; orde-1 = bias global +
    bias user + bias item; deep = MLP atas [v_user; v_item]. Output sigmoid."""

    def __init__(self, n_users: int, n_items: int, config: NeuralCFConfig):
        super().__init__()
        d = config.embedding_dim
        # Embedding orde-2 (dibagi FM & deep, sesuai paper).
        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        # Orde-1 (linear): bobot skalar per user/item.
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        layers = []
        input_dim = 2 * d
        for hidden_dim in config.mlp_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        v_u = self.user_emb(user_idx)
        v_i = self.item_emb(item_idx)
        # Orde-1 linear.
        first_order = (
            self.global_bias
            + self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
        )
        # Orde-2 FM (2 field) = <v_u, v_i>.
        second_order = (v_u * v_i).sum(dim=-1)
        # Komponen deep.
        deep_out = self.deep(torch.cat([v_u, v_i], dim=-1)).squeeze(-1)
        logit = first_order + second_order + deep_out
        return torch.sigmoid(logit)


class NeuralCFTrainer:
    """Wrapper fit/predict untuk NeuMF atau DeepFM, API konsisten dengan
    Trainer lain di repo ini (fit(train_df, val_df) -> predict(df))."""

    def __init__(self, n_users: int, n_items: int, config: NeuralCFConfig | None = None):
        self.config = config or NeuralCFConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if self.config.model == "neumf":
            self.model = NeuMFModel(n_users, n_items, self.config).to(self.config.device)
        elif self.config.model == "deepfm":
            self.model = DeepFMModel(n_users, n_items, self.config).to(self.config.device)
        else:
            raise ValueError(
                f"model '{self.config.model}' tidak dikenal -- gunakan 'neumf' atau 'deepfm'"
            )

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        rating_scale: tuple[float, float] = (1.0, 5.0),
    ) -> None:
        # Mapping user/item dari TRAIN saja (cold-start test ditangani predict).
        self.user2idx = {u: i for i, u in enumerate(train_df["user_id"].unique())}
        self.item2idx = {b: i for i, b in enumerate(train_df["business_id"].unique())}
        self.rating_scale = rating_scale

        train_ds = RatingDataset(train_df, self.user2idx, self.item2idx, rating_scale)
        loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()

        val_ds = None
        if val_df is not None:
            # Buang baris val dgn user/item tak dikenal (tak bisa dievaluasi model).
            mask = val_df["user_id"].isin(self.user2idx) & val_df["business_id"].isin(self.item2idx)
            if mask.any():
                val_ds = RatingDataset(val_df[mask], self.user2idx, self.item2idx, rating_scale)

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
            log_msg = f"[{self.config.model}] Epoch {epoch + 1}/{self.config.epochs} - train MSE: {avg_loss:.4f}"
            if val_ds is not None:
                val_rmse = self._evaluate_rmse_norm(val_ds)
                log_msg += f" - val RMSE (normalized): {val_rmse:.4f}"
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    log_msg += " (terbaik, disimpan)"
            logger.info(log_msg)

        if val_ds is not None and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(
                "Restore bobot %s dari epoch val RMSE terbaik (%.4f), bukan epoch terakhir.",
                self.config.model, best_val_rmse,
            )

    @torch.no_grad()
    def _evaluate_rmse_norm(self, dataset: RatingDataset) -> float:
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
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Prediksi rating skala asli 1-5. Cold-start (user/item baru) fallback
        ke rating rata-rata skala, konsisten dengan deepmf.py/classical_cf.py."""
        self.model.eval()
        rating_min, rating_max = self.rating_scale
        scale_range = rating_max - rating_min

        user_idx = df["user_id"].map(self.user2idx)
        item_idx = df["business_id"].map(self.item2idx)
        unknown_mask = user_idx.isna() | item_idx.isna()

        if unknown_mask.any():
            logger.info(
                "%d/%d baris memakai fallback rata-rata skala (%.2f) (user/item baru) saat prediksi %s",
                int(unknown_mask.sum()), len(df), rating_min + scale_range / 2, self.config.model,
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
                p = self.model(u_tensor[start:start + batch_size], i_tensor[start:start + batch_size])
                known_preds.append(p.cpu().numpy())
            preds[known_rows] = np.concatenate(known_preds) * scale_range + rating_min

        self.model.train()
        return np.clip(preds, rating_min, rating_max)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton neural_cf (NeuMF/DeepFM) -- jalankan via run_neural_cf.py "
        "setelah split tersedia (data/splits/<domain>/)."
    )
