"""VoltaCast's main model: a seq2seq Transformer for quantile forecasting.

Encoder reads one week of hourly history (load + covariates). Decoder queries
are the 24 future hours, represented by their known covariates (calendar +
weather forecast); each query cross-attends over the encoded history. The
forecast is non-autoregressive: all 24 hours and all quantiles come out in a
single forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .heads import monotone_quantiles


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return x + self.pe[: x.size(1)]


class VoltaCastTransformer(nn.Module):
    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        self.quantiles = quantiles
        self.past_proj = nn.Linear(n_past_features, d_model)
        self.future_proj = nn.Linear(n_future_features, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, enc_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.head = nn.Linear(d_model, len(quantiles))

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """x_past (B, C, Fp), x_future (B, H, Ff) → (B, H, Q) quantile forecasts."""
        memory = self.encoder(self.pos_enc(self.past_proj(x_past)))
        queries = self.pos_enc(self.future_proj(x_future))
        decoded = self.decoder(queries, memory)
        return monotone_quantiles(self.head(decoded))
