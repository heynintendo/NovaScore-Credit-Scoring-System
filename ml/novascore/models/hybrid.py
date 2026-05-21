"""HybridModel — fuses FT-Transformer (tabular) + TCN (weekly sequence) + Node2Vec (graph).

Returns raw logits; pair with BCEWithLogitsLoss during training and apply
torch.sigmoid for probability outputs at inference time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ft_transformer import FTTransformer
from .tcn import TCNEncoder


class HybridModel(nn.Module):
    """Three-tower fusion model.

    Args:
        n_tab: number of tabular features fed to FT-Transformer.
        d_tab: FT-Transformer hidden dim.
        d_seq: TCN hidden dim.
        g_dim: Node2Vec embedding dim (passed through as-is).
        n_seq_features: per-week feature count for TCN input.
        dropout: fusion-head dropout probability.
    """

    def __init__(
        self,
        n_tab: int,
        d_tab: int = 256,
        d_seq: int = 128,
        g_dim: int = 64,
        n_seq_features: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ft = FTTransformer(n_num=n_tab, d_model=d_tab)
        self.tcn = TCNEncoder(in_dim=n_seq_features, d_model=d_seq)
        # Node2Vec vectors are already learned offline; just pass through.
        self.graph = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(d_tab + d_seq + g_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_seq: torch.Tensor,
        x_g: torch.Tensor,
    ) -> torch.Tensor:
        h_tab = self.ft(x_tab)
        h_seq = self.tcn(x_seq)
        h_g = self.graph(x_g)
        h = torch.cat([h_tab, h_seq, h_g], dim=-1)
        return self.head(h).squeeze(-1)
