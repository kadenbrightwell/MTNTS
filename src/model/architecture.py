"""Neural network architectures for multi-ticker return prediction.

Each target ticker (UVXY, SPXU, SVIX, SPXL) gets its own model instance.
Two architectures selectable via ModelConfig.model_type:
  - "lstm"        : Bidirectional LSTM with multi-head self-attention
  - "transformer" : Temporal Transformer encoder
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ModelConfig, DEVICE


# ---------------------------------------------------------------------------
# Positional encoding (shared by transformer)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# LSTM with multi-head self-attention
# ---------------------------------------------------------------------------

class LSTMAttentionModel(nn.Module):
    """Bidirectional LSTM followed by multi-head self-attention and FC head."""

    def __init__(self, n_features: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        self.layer_norm = nn.LayerNorm(n_features)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        lstm_out_size = cfg.hidden_size * 2  # bidirectional
        self.lstm_drop = nn.Dropout(cfg.dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, cfg.fc_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.layer_norm(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = self.lstm_drop(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        pooled = attn_out[:, -1, :]  # last timestep
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformerModel(nn.Module):
    """Transformer encoder for time-series return prediction."""

    def __init__(self, n_features: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        d_model = cfg.hidden_size
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=cfg.seq_len + 64, dropout=cfg.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)  # global average pooling
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(n_features: int, cfg: ModelConfig | None = None) -> nn.Module:
    """Instantiate the configured model on the target device."""
    cfg = cfg or ModelConfig()
    if cfg.model_type == "transformer":
        model = TemporalTransformerModel(n_features, cfg)
    else:
        model = LSTMAttentionModel(n_features, cfg)

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {cfg.model_type.upper()} on {DEVICE}  |  "
          f"params: {total_params:,} total, {trainable:,} trainable")
    return model
