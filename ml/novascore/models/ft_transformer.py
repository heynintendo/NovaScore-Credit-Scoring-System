"""FT-Transformer for tabular features with per-feature tokenization.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).
Each scalar feature is projected to its own d_model-dim token; a learned CLS token
is concatenated and the sequence is run through a small TransformerEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Project each scalar feature x_i to a token w_i * x_i + b_i in R^{d_model}."""

    def __init__(self, n_num: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_model))
        self.bias = nn.Parameter(torch.zeros(n_num, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        tok = x_num.unsqueeze(-1) * self.weight + self.bias
        return self.ln(tok)


class FTTransformer(nn.Module):
    """Stack of TransformerEncoder layers over per-feature tokens; pooled by CLS."""

    def __init__(
        self,
        n_num: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tok = FeatureTokenizer(n_num, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        B = x_num.size(0)
        x = self.tok(x_num)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        h = self.encoder(x)
        return self.out_ln(h[:, 0, :])
