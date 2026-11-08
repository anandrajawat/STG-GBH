"""
Multi-Task prediction heads and uncertainty-weighted loss balancing for STG-GBH.
Heads:
  1. Binary intrusion detection
  2. Attack category classification
  3. Attack-stage estimation (MITRE ATT&CK tactical stages)
  4. Future attack forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiTaskPredictionHeads(nn.Module):
    """
    Modular multi-task output heads branching from the fused representation.
    """
    def __init__(
        self,
        in_dim: int,
        num_categories: int = 11,
        num_mitre_stages: int = 6
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_categories = num_categories
        self.num_mitre_stages = num_mitre_stages

        # Head 1: Binary intrusion detection
        self.binary_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 2)
        )

        # Head 2: Attack category classification
        self.category_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_categories)
        )

        # Head 3: MITRE ATT&CK stage estimation
        self.mitre_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_mitre_stages)
        )

        # Head 4: Future attack activity forecasting (binary next-window indicator)
        self.future_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "binary_logits": self.binary_head(z),
            "category_logits": self.category_head(z),
            "mitre_logits": self.mitre_head(z),
            "future_logits": self.future_head(z).squeeze(-1)
        }


class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall & Gal homoscedastic uncertainty loss balancing.
    Learns task log-variances: L_total = sum_i ( 0.5 * exp(-s_i) * L_i + 0.5 * s_i )
    where s_i = log(sigma_i^2).
    """
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        # Initialize log_vars to 0.0 (sigma = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        binary_loss: torch.Tensor,
        cat_loss: Optional[torch.Tensor] = None,
        mitre_loss: Optional[torch.Tensor] = None,
        future_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = [binary_loss]
        if cat_loss is not None:
            losses.append(cat_loss)
        if mitre_loss is not None:
            losses.append(mitre_loss)
        if future_loss is not None:
            losses.append(future_loss)

        total_loss = 0.0
        log_dict = {}

        for i, l in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            task_l = 0.5 * precision * l + 0.5 * self.log_vars[i]
            total_loss += task_l
            log_dict[f"task_{i}_sigma"] = float(torch.exp(0.5 * self.log_vars[i]).item())
            log_dict[f"task_{i}_raw_loss"] = float(l.item())

        return total_loss, log_dict

