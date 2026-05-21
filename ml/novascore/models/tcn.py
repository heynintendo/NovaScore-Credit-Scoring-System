"""Temporal Convolutional Network encoder over weekly sequence features.

Reference: Bai, Kolter, Koltun, "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" (2018). Stacked dilated causal
Conv1d blocks with residual connections; output mean-pooled across time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Trim trailing pad positions so causal Conv1d preserves input length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two causal dilated Conv1d layers + residual + ReLU."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.res(x))


class TCNEncoder(nn.Module):
    """3 dilated blocks (dilations 1,2,4) with AdaptiveAvgPool over time."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_blocks: int = 3,
        kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ch = in_dim
        layers: list[nn.Module] = []
        for b in range(n_blocks):
            layers.append(
                TemporalBlock(ch, d_model, kernel, dilation=2**b, dropout=dropout)
            )
            ch = d_model
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F) -> (B, F, T)
        x = x_seq.transpose(1, 2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.ln(h)
