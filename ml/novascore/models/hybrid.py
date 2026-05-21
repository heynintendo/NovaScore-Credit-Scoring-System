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
        g_dim: Node2Vec embedding dim (passed through as-is). Set to 0 to omit
            the graph tower entirely — the production model after the Phase 4.5
            ablation uses g_dim=0.
        n_seq_features: per-week feature count for TCN input.
        n_layers: number of FT-Transformer encoder layers.
        dropout: fusion-head dropout probability.
    """

    def __init__(
        self,
        n_tab: int,
        d_tab: int = 256,
        d_seq: int = 128,
        g_dim: int = 0,
        n_seq_features: int = 9,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.g_dim = g_dim
        self.ft = FTTransformer(n_num=n_tab, d_model=d_tab, n_layers=n_layers, dropout=dropout)
        self.tcn = TCNEncoder(in_dim=n_seq_features, d_model=d_seq, dropout=dropout)
        # Node2Vec vectors are already learned offline; identity is just for symmetry.
        self.graph = nn.Identity()
        head_in = d_tab + d_seq + max(0, g_dim)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
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
        if self.g_dim > 0:
            h_g = self.graph(x_g)
            h = torch.cat([h_tab, h_seq, h_g], dim=-1)
        else:
            h = torch.cat([h_tab, h_seq], dim=-1)
        return self.head(h).squeeze(-1)
