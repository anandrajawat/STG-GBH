"""
Temporal Attention module for STG-GBH.
Supports Additive Attention (Bahdanau-style), Scaled Dot-Product Attention,
Mean Pooling, and Max Pooling across historical temporal snapshots with
residual preservation of the current target timestep.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """
    Temporal aggregator over historical hidden states (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, hidden_dim: int = 128, mode: str = "additive"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode.lower()

        if self.mode == "additive":
            self.w_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, 1, bias=False)
            self.out_ln = nn.LayerNorm(hidden_dim)
        elif self.mode == "scaled_dot":
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.scale = 1.0 / math.sqrt(hidden_dim)
            self.out_ln = nn.LayerNorm(hidden_dim)
        elif self.mode in ["mean", "max"]:
            self.out_ln = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown temporal attention mode: '{mode}'. Choose from ['additive', 'scaled_dot', 'mean', 'max']")

    def forward(self, h_seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_seq: (batch_size, seq_len, hidden_dim)
            mask: optional boolean mask (batch_size, seq_len) where False/0 indicates padding.
        Returns:
            context: (batch_size, hidden_dim) aggregated temporal representation
            weights: (batch_size, seq_len) attention weights (or uniform weights for pooling)
        """
        batch_size, seq_len, _ = h_seq.shape
        h_last = h_seq[:, -1, :]

        if self.mode == "mean":
            context = self.out_ln(0.5 * torch.mean(h_seq, dim=1) + 0.5 * h_last)
            weights = torch.full((batch_size, seq_len), 1.0 / seq_len, device=h_seq.device)
            return context, weights

        if self.mode == "max":
            max_pool, _ = torch.max(h_seq, dim=1)
            context = self.out_ln(0.5 * max_pool + 0.5 * h_last)
            weights = torch.full((batch_size, seq_len), 1.0 / seq_len, device=h_seq.device)
            return context, weights

        if self.mode == "additive":
            score = self.v_proj(torch.tanh(self.w_proj(h_seq))).squeeze(-1)
        elif self.mode == "scaled_dot":
            q = self.query.expand(batch_size, -1, -1)
            score = torch.bmm(q, h_seq.transpose(1, 2)).squeeze(1) * self.scale

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        weights = F.softmax(score, dim=-1)
        attn_pool = torch.bmm(weights.unsqueeze(1), h_seq).squeeze(1)
        context = self.out_ln(0.5 * attn_pool + 0.5 * h_last)

        return context, weights
