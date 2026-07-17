"""
src/a2fusionrs/attention_gated_fusion.py

Modul Attention-Gated Fusion (A2-FusionRS Fase 2) -- MENGGANTIKAN fusi
statis NMF+DecisionTreeRegressor (Fase 1, src/baseline/fusion_nmf_dt.py)
dengan lapisan neural kecil yang menggabungkan representasi VEKTOR dari
modalitas (DeepMF laten, fitur CBF, skor ABSA) via cross-attention + gated
fusion, alih-alih hanya skalar prediksi akhir seperti Fase 1. Lihat
phase2_notes/attention_gated_fusion_design.md untuk desain lengkap & alasan
arsitektur, serta cakupan eksperimen 3-tier yang memakai varian di bawah.

SATU kelas (`AttentionGatedFusionModel`) melayani BEBERAPA varian ablasi
Tier 1/2 via kombinasi 2 flag config, TANPA duplikasi kelas per varian:
- `use_attention=True,  pooling="gate"`   -> Full AGF (varian utama)
- `use_attention=True,  pooling="mean"`   -> Attention-only (tanpa gate)
- `use_attention=False, pooling="gate"`   -> Gating-only (tanpa attention)
- `use_attention=False, pooling="concat"` -> baseline eksternal "Concat+MLP"
  (deep, tanpa attention maupun gating -- pembanding Tier 2)
Leave-one-modality-out (mis. tanpa CBF) otomatis didukung TANPA kode
tambahan -- cukup kirim dict `features` dgn subset modalitas yang lebih
sedikit ke fit()/predict(), proyeksi & attention menyesuaikan jumlah token
secara dinamis mengikuti jumlah key di dict tsb.

Baseline "Weighted-average tetap" (Tier 2, gating naif TANPA jaringan
neural sama sekali) SENGAJA TIDAK dimasukkan ke modul ini -- itu beroperasi
di ruang SKALAR (prediksi akhir DeepMF/CBF/ABSA), bukan ruang vektor seperti
modul ini, dan cukup diimplementasikan langsung di
`run_attention_gated_fusion.py` (Stage 3) sbg regresi linear sederhana.

Skema training: 2 tahap (lihat design doc Bagian 2) -- DeepMF/CBF/ABSA
SUDAH dilatih/di-fit/di-skor secara terpisah (expert beku), modul ini
HANYA melatih lapisan fusi kecil di atas fitur yang sudah diekstrak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AGFConfig:
    d: int = 64  # dimensi embedding bersama antar modalitas
    n_heads: int = 2
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0  # L2 regularisasi Adam -- utk mode residual,
    # weight_decay>0 menekan koreksi ke arah nol (belajar hanya koreksi
    # robust, bukan overfit noise train). Default 0 = perilaku lama.
    use_attention: bool = True
    pooling: str = "gate"  # "gate" | "mean" | "concat"
    # residual: kalau True, model memprediksi KOREKSI di atas prediksi base
    # (base + head(fused), TANPA sigmoid) alih-alih prediksi absolut
    # (sigmoid(head(fused))). Alasan (Stage 7 root-cause): DecisionTree
    # A2-IRM sulit dikalahkan dari nol; struktur residual membuat AGF
    # "base + koreksi adaptif" -- kalau koreksi=0 menyamai base, koreksi yg
    # membantu MENGALAHKAN base. base diberikan per-baris via fit()/predict().
    residual: bool = False
    # aspect_pooling (Jalur X): kalau True, tambah 1 token modalitas dari
    # AspectSequencePooling atas sequence aspek PyABSA panjang-variabel
    # (embedding IDENTITAS aspek + sentimen per-aspek) -- sesuatu yg tree
    # tak bisa konsumsi, satu-satunya keunggulan struktural AGF atas tree.
    aspect_pooling: bool = False
    aspect_vocab_size: int = 0     # jumlah istilah aspek unik (utk nn.Embedding)
    aspect_emb_dim: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.pooling not in ("gate", "mean", "concat"):
            raise ValueError(f"pooling '{self.pooling}' tidak dikenal -- pakai 'gate'/'mean'/'concat'.")
        if self.use_attention and self.d % self.n_heads != 0:
            raise ValueError(
                f"d ({self.d}) harus habis dibagi n_heads ({self.n_heads}) -- "
                "syarat nn.MultiheadAttention."
            )


class AspectSequencePooling(nn.Module):
    """Attention-pooling atas sequence aspek PyABSA panjang-variabel (Jalur X).
    Tiap aspek = [embedding_identitas_aspek, P_neg, P_neu, P_pos, confidence].
    Query terlatih meng-attend seluruh aspek (mask padding) -> 1 vektor d.
    Ini yg tree TAK BISA lakukan (sequence var-length + identitas aspek);
    kalau identitas aspek mengandung sinyal, di sinilah AGF unggul unik."""

    def __init__(self, vocab_size: int, aspect_emb_dim: int, d: int, n_heads: int, dropout: float):
        super().__init__()
        # +2: id 0=PAD (padding_idx), 1=UNK
        self.aspect_emb = nn.Embedding(vocab_size + 2, aspect_emb_dim, padding_idx=0)
        self.token_proj = nn.Linear(aspect_emb_dim + 4, d)  # emb + [P_neg,P_neu,P_pos,conf]
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids (B,L) long, feats (B,L,4), mask (B,L) bool True=valid
        emb = self.aspect_emb(ids)                                   # (B,L,emb)
        tokens = self.token_proj(torch.cat([emb, feats], dim=-1))    # (B,L,d)
        q = self.query.expand(tokens.shape[0], -1, -1)               # (B,1,d)
        pooled, _ = self.attn(q, tokens, tokens, key_padding_mask=~mask)  # (B,1,d)
        return pooled.squeeze(1)                                     # (B,d)


class AttentionGatedFusionModel(nn.Module):
    """`modality_dims`: dict {nama_modalitas: dimensi_input_mentah} --
    menentukan proyeksi linear per modalitas. Urutan key dict ini menentukan
    urutan token pada sequence attention (HARUS konsisten antara fit() dan
    predict() -- dijamin oleh AttentionGatedFusionTrainer yang menyimpan
    urutan ini sekali di __init__, bukan dibaca ulang tiap panggilan)."""

    def __init__(self, modality_dims: dict[str, int], config: AGFConfig):
        super().__init__()
        self.config = config
        self.modalities = list(modality_dims.keys())

        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, config.d) for name, dim in modality_dims.items()}
        )

        # aspect_pooling (Jalur X): 1 token EKSTRA dari sequence aspek PyABSA.
        if config.aspect_pooling:
            self.aspect_pooling = AspectSequencePooling(
                config.aspect_vocab_size, config.aspect_emb_dim, config.d, config.n_heads, config.dropout
            )
        # jumlah token total (modalitas fixed + 1 token aspek kalau aktif) --
        # menentukan dimensi gate_net & concat head.
        self.n_tokens = len(self.modalities) + (1 if config.aspect_pooling else 0)

        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.d, num_heads=config.n_heads, dropout=config.dropout, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(config.d)

        if config.pooling == "gate":
            self.gate_net = nn.Sequential(
                nn.Linear(config.d * self.n_tokens, config.d),
                nn.ReLU(),
                nn.Linear(config.d, self.n_tokens),
            )

        head_input_dim = config.d * self.n_tokens if config.pooling == "concat" else config.d
        self.prediction_head = nn.Sequential(
            nn.Linear(head_input_dim, max(config.d // 2, 1)),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(max(config.d // 2, 1), 1),
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        base: torch.Tensor | None = None,
        aspect_seq: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # `base`: prediksi base ternormalisasi (batch,) -- WAJIB kalau residual.
        # `aspect_seq`: dict {ids,feats,mask} sequence aspek -- WAJIB kalau
        # config.aspect_pooling. Token aspek DITEMPEL setelah token modalitas.
        # Proyeksi tiap modalitas ke dimensi bersama d, susun jadi sequence.
        token_list = [self.projections[name](features[name]) for name in self.modalities]
        if self.config.aspect_pooling:
            if aspect_seq is None:
                raise ValueError("config.aspect_pooling=True tapi aspect_seq tidak diberikan ke forward().")
            token_list.append(self.aspect_pooling(aspect_seq["ids"], aspect_seq["feats"], aspect_seq["mask"]))
        tokens = torch.stack(token_list, dim=1)  # (batch, n_tokens, d)

        attn_weights = None
        if self.config.use_attention:
            attended, attn_weights = self.attention(tokens, tokens, tokens, need_weights=True)
            tokens = self.attn_norm(tokens + attended)  # residual + LayerNorm

        gate_weights = None
        if self.config.pooling == "concat":
            fused = tokens.reshape(tokens.shape[0], -1)
        elif self.config.pooling == "mean":
            fused = tokens.mean(dim=1)
        else:  # "gate"
            flat = tokens.reshape(tokens.shape[0], -1)
            gate_logits = self.gate_net(flat)
            gate_weights = torch.softmax(gate_logits, dim=-1)  # (batch, n_mod), jumlah=1 per baris
            fused = (tokens * gate_weights.unsqueeze(-1)).sum(dim=1)  # (batch, d)

        head_out = self.prediction_head(fused).squeeze(-1)
        if self.config.residual:
            if base is None:
                raise ValueError("config.residual=True tapi `base` tidak diberikan ke forward().")
            # KOREKSI aditif (tanpa sigmoid) di atas base -- residual bisa +/-.
            pred = base + head_out
        else:
            pred = torch.sigmoid(head_out)
        return pred, gate_weights, attn_weights


class AttentionGatedFusionTrainer:
    """Interface fit()/predict() semirip DeepMFTrainer (src/baseline/deepmf.py)
    -- termasuk pola "lacak & restore state_dict dgn val RMSE terbaik", sama
    seperti DeepMF (model kecil ini juga berpotensi overfit tanpa early
    stopping eksplisit)."""

    def __init__(self, modality_dims: dict[str, int], config: AGFConfig | None = None):
        self.config = config or AGFConfig()
        self.modality_dims = dict(modality_dims)
        self.model = AttentionGatedFusionModel(self.modality_dims, self.config).to(self.config.device)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _aspect_to_tensors(self, aspect_seq: dict[str, np.ndarray] | None) -> dict[str, torch.Tensor] | None:
        """Konversi {ids,feats,mask} numpy -> tensor (dtype benar: ids long,
        mask bool). None kalau tidak ada."""
        if aspect_seq is None:
            return None
        dev = self.config.device
        return {
            "ids": torch.tensor(aspect_seq["ids"], dtype=torch.long, device=dev),
            "feats": torch.tensor(aspect_seq["feats"], dtype=torch.float32, device=dev),
            "mask": torch.tensor(aspect_seq["mask"], dtype=torch.bool, device=dev),
        }

    def _to_tensors(self, features: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        missing = set(self.modality_dims) - set(features)
        if missing:
            raise KeyError(
                f"Modalitas {missing} ada di modality_dims (saat __init__) tapi hilang dari "
                "features yang diberikan -- fit()/predict() HARUS memakai set modalitas yang sama persis."
            )
        return {
            name: torch.tensor(features[name], dtype=torch.float32, device=self.config.device)
            for name in self.modality_dims
        }

    def fit(
        self,
        train_features: dict[str, np.ndarray],
        train_ratings_norm: np.ndarray,
        val_features: dict[str, np.ndarray] | None = None,
        val_ratings_norm: np.ndarray | None = None,
        train_base_norm: np.ndarray | None = None,
        val_base_norm: np.ndarray | None = None,
        train_aspect_seq: dict[str, np.ndarray] | None = None,
        val_aspect_seq: dict[str, np.ndarray] | None = None,
    ) -> float:
        """`*_ratings_norm` HARUS sudah dinormalisasi ke (0,1) (konvensi sama
        dgn DeepMF/InteractionDataset) -- denormalisasi ke skala rating asli
        terjadi di predict(), bukan di sini.

        `*_base_norm`: prediksi base ternormalisasi (WAJIB kalau
        config.residual=True) -- model belajar KOREKSI di atas base ini.

        Return: waktu training (detik) -- dicatat di sini (bukan di caller)
        supaya konsisten dipakai semua pemanggil untuk Tabel efisiensi
        Tier 3 (lihat attention_gated_fusion_design.md Bagian 3, poin 8).
        """
        if self.config.residual and train_base_norm is None:
            raise ValueError("config.residual=True tapi train_base_norm tidak diberikan ke fit().")
        if self.config.aspect_pooling and train_aspect_seq is None:
            raise ValueError("config.aspect_pooling=True tapi train_aspect_seq tidak diberikan ke fit().")
        t0 = time.time()
        train_tensors = self._to_tensors(train_features)
        y_tensor = torch.tensor(train_ratings_norm, dtype=torch.float32, device=self.config.device)
        base_tensor = (
            torch.tensor(train_base_norm, dtype=torch.float32, device=self.config.device)
            if train_base_norm is not None else None
        )
        aspect_tensors = self._aspect_to_tensors(train_aspect_seq)
        n_samples = len(train_ratings_norm)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")
        best_state_dict = None

        self.model.train()
        for epoch in range(self.config.epochs):
            perm = torch.randperm(n_samples, device=self.config.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                batch_features = {name: t[idx] for name, t in train_tensors.items()}
                batch_y = y_tensor[idx]
                batch_base = base_tensor[idx] if base_tensor is not None else None
                batch_aspect = (
                    {k: v[idx] for k, v in aspect_tensors.items()} if aspect_tensors is not None else None
                )

                optimizer.zero_grad()
                pred, _, _ = self.model(batch_features, base=batch_base, aspect_seq=batch_aspect)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            log_msg = f"AGF epoch {epoch + 1}/{self.config.epochs} - train MSE: {epoch_loss / n_batches:.4f}"
            if val_features is not None and val_ratings_norm is not None:
                val_rmse = self.evaluate_rmse(val_features, val_ratings_norm, val_base_norm, val_aspect_seq)
                log_msg += f" - val RMSE (normalized): {val_rmse:.4f}"
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    log_msg += " (terbaik sejauh ini, disimpan)"
            logger.info(log_msg)

        if val_features is not None and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restore bobot AGF dari epoch dengan val RMSE terbaik (%.4f).", best_val_rmse)

        train_time = time.time() - t0
        logger.info(
            "Training AGF selesai dalam %.1f detik (%d parameter trainable).",
            train_time,
            self.n_parameters,
        )
        return train_time

    @torch.no_grad()
    def evaluate_rmse(
        self, features: dict[str, np.ndarray], ratings_norm: np.ndarray, base_norm: np.ndarray | None = None,
        aspect_seq: dict[str, np.ndarray] | None = None,
    ) -> float:
        self.model.eval()
        tensors = self._to_tensors(features)
        y_tensor = torch.tensor(ratings_norm, dtype=torch.float32, device=self.config.device)
        base_tensor = (
            torch.tensor(base_norm, dtype=torch.float32, device=self.config.device)
            if base_norm is not None else None
        )
        pred, _, _ = self.model(tensors, base=base_tensor, aspect_seq=self._aspect_to_tensors(aspect_seq))
        rmse = torch.sqrt(torch.mean((pred - y_tensor) ** 2)).item()
        self.model.train()
        return rmse

    @torch.no_grad()
    def predict(
        self,
        features: dict[str, np.ndarray],
        rating_scale: tuple[float, float] = (1.0, 5.0),
        base_norm: np.ndarray | None = None,
        aspect_seq: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (prediksi rating skala asli, bobot gate per-baris -- None
        kalau pooling != 'gate'). `base_norm` WAJIB kalau config.residual=True,
        `aspect_seq` WAJIB kalau config.aspect_pooling=True."""
        if self.config.residual and base_norm is None:
            raise ValueError("config.residual=True tapi base_norm tidak diberikan ke predict().")
        t0 = time.time()
        self.model.eval()
        rating_min, rating_max = rating_scale
        scale_range = rating_max - rating_min

        tensors = self._to_tensors(features)
        base_tensor = (
            torch.tensor(base_norm, dtype=torch.float32, device=self.config.device)
            if base_norm is not None else None
        )
        pred, gate_weights, _ = self.model(
            tensors, base=base_tensor, aspect_seq=self._aspect_to_tensors(aspect_seq)
        )

        preds = pred.cpu().numpy() * scale_range + rating_min
        gates = gate_weights.cpu().numpy() if gate_weights is not None else None
        self.model.train()

        predict_time = time.time() - t0
        n_rows = len(preds)
        logger.info(
            "Prediksi AGF selesai dalam %.3f detik (%d baris, %.5f detik/baris).",
            predict_time,
            n_rows,
            predict_time / n_rows if n_rows else 0.0,
        )
        return preds, gates


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Skeleton Attention-Gated Fusion -- jalankan via run_attention_gated_fusion.py, "
        "jangan berdiri sendiri."
    )
