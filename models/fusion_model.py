"""
Graph and Boosting Fusion Model (STG-GBH).
Combines Spatio-Temporal Graph Embeddings with OOF Boosting Probabilities & Log-Odds:
z_t = [h_graph || p_boost || logit(p_boost)]
Integrates a log-odds boosting skip connection with a bounded tanh spatio-temporal
graph residual correction so high-confidence boosting consensus is preserved while
ambiguous boundary flows are resolved via spatio-temporal graph context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .multitask_heads import MultiTaskPredictionHeads


class STGGBHFusionModel(nn.Module):
    """
    Learnable Bounded-Residual Graph-and-Boosting Fusion Head for STG-GBH.
    """
    def __init__(
        self,
        graph_embed_dim: int = 128,
        num_boosting_prob_cols: int = 4,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.15,
        enable_multitask: bool = True,
        num_categories: int = 11,
        num_mitre_stages: int = 6
    ):
        super().__init__()
        self.graph_embed_dim = graph_embed_dim
        self.num_boosting_prob_cols = num_boosting_prob_cols
        self.in_fusion_dim = graph_embed_dim + 2 * num_boosting_prob_cols
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.enable_multitask = enable_multitask

        # Dedicated boosting representation projector
        self.boost_proj = nn.Sequential(
            nn.Linear(2 * num_boosting_prob_cols, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # Fusion MLP over [h_graph || p_boost || logit(p_boost)]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.in_fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Residual skip connection from boosting log-odds directly to output logits
        self.boost_skip = nn.Linear(num_boosting_prob_cols, num_classes)
        with torch.no_grad():
            if num_classes == 2:
                if num_boosting_prob_cols == 4:
                    # Default ensemble prior (XGB, CatBoost, LightGBM, OOF-Stack)
                    w = torch.tensor([0.15, 0.50, 0.10, 0.25], dtype=torch.float32)
                    self.boost_skip.weight[0].copy_(-0.5 * w)
                    self.boost_skip.weight[1].copy_(0.5 * w)
                else:
                    w = 1.0 / max(1, num_boosting_prob_cols)
                    self.boost_skip.weight[0].fill_(-0.5 * w)
                    self.boost_skip.weight[1].fill_(0.5 * w)
                self.boost_skip.bias.zero_()

        # Single-task primary output head (learns bounded spatio-temporal residual correction)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.res_scale = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        with torch.no_grad():
            self.classifier.weight.mul_(0.05)
            self.classifier.bias.zero_()

        # Multi-task heads
        if enable_multitask:
            self.multitask_heads = MultiTaskPredictionHeads(
                in_dim=hidden_dim,
                num_categories=num_categories,
                num_mitre_stages=num_mitre_stages
            )
        else:
            self.multitask_heads = None

    def forward(
        self,
        h_graph: torch.Tensor,
        p_boosting: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Args:
            h_graph: (batch_size, graph_embed_dim) spatio-temporal graph representation
            p_boosting: (batch_size, num_boosting_prob_cols) boosting probability features
        Returns:
            logits: (batch_size, num_classes) primary task logits
            mt_outputs: dict of multi-task logits (or None)
            z: (batch_size, hidden_dim) fused latent representation
        """
        p_clamped = torch.clamp(p_boosting, 1e-5, 1.0 - 1e-5)
        boost_logits = torch.log(p_clamped / (1.0 - p_clamped))

        z_raw = torch.cat([h_graph, p_clamped, boost_logits], dim=-1)
        z = self.fusion_mlp(z_raw)

        scale = torch.clamp(self.res_scale, 0.1, 1.2)
        logits = self.boost_skip(boost_logits) + scale * torch.tanh(self.classifier(z))
        mt_outputs = self.multitask_heads(z) if self.multitask_heads is not None else None

        return logits, mt_outputs, z

    def predict_proba(self, h_graph: torch.Tensor, p_boosting: torch.Tensor) -> torch.Tensor:
        """Computes softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(h_graph, p_boosting)
            probas = F.softmax(logits, dim=-1)
        return probas
