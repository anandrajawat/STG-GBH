"""
Temporal GRU branch for STG-GBH.
Processes historical sequences of graph embeddings in strict causal unidirectional order.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TemporalGRU(nn.Module):
    """
    Unidirectional GRU for historical graph embedding sequences.
    Strictly causal (no bidirectional lookahead).
    """
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_seq: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: (batch_size, seq_len, embed_dim) historical sequence tensor.
            h_0: (num_layers, batch_size, hidden_dim) optional initial hidden state.
        Returns:
            outputs: (batch_size, seq_len, hidden_dim)
            h_n: (num_layers, batch_size, hidden_dim)
        """
        gru_out, h_n = self.gru(x_seq, h_0)
        gru_out = self.dropout(self.ln(gru_out))
        return gru_out, h_n

