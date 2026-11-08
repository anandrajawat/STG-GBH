"""
Inductive Heterogeneous GraphSAGE Encoder for STG-GBH.
Propagates multi-relational neighborhood context to flow nodes with arcsinh input
stabilization and residual skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from typing import Dict, Tuple


class HeteroGraphSAGE(nn.Module):
    """
    Two-layer inductive GraphSAGE architecture with LayerNorm, ReLU, and L2 normalization.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        dropout: float = 0.2,
        aggr: str = "mean"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout

        self.flow_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv1 = HeteroConv({
            ("src_ip", "COMMUNICATES_WITH", "dst_ip"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr),
            ("src_ip", "INITIATES_FLOW", "flow"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr),
            ("flow", "TARGETS", "dst_ip"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr),
            ("flow", "USES_SERVICE", "service"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr),
            ("flow", "USES_PROTOCOL", "protocol"): SAGEConv(hidden_dim, hidden_dim, aggr=aggr),
        }, aggr="sum")

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.conv2 = HeteroConv({
            ("src_ip", "COMMUNICATES_WITH", "dst_ip"): SAGEConv(hidden_dim, out_dim, aggr=aggr),
            ("src_ip", "INITIATES_FLOW", "flow"): SAGEConv(hidden_dim, out_dim, aggr=aggr),
            ("flow", "TARGETS", "dst_ip"): SAGEConv(hidden_dim, out_dim, aggr=aggr),
            ("flow", "USES_SERVICE", "service"): SAGEConv(hidden_dim, out_dim, aggr=aggr),
            ("flow", "USES_PROTOCOL", "protocol"): SAGEConv(hidden_dim, out_dim, aggr=aggr),
        }, aggr="sum")

        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        h_dict = {}
        for node_type, x in x_dict.items():
            x_clean = torch.asinh(torch.clamp(x, -1e6, 1e6))
            if node_type == "flow":
                h_dict[node_type] = self.flow_proj(x_clean)
            else:
                h_dict[node_type] = self.node_proj(x_clean)

        conv1_out = self.conv1(h_dict, edge_index_dict)
        for node_type in h_dict:
            if node_type in conv1_out:
                h_dict[node_type] = h_dict[node_type] + F.relu(self.ln1(conv1_out[node_type]))
                h_dict[node_type] = F.dropout(h_dict[node_type], p=self.dropout_rate, training=self.training)

        conv2_out = self.conv2(h_dict, edge_index_dict)
        out_dict = {}
        for node_type in h_dict:
            if node_type in conv2_out:
                h = h_dict[node_type] + self.ln2(conv2_out[node_type])
                out_dict[node_type] = F.normalize(h, p=2, dim=-1)
            else:
                out_dict[node_type] = F.normalize(h_dict[node_type], p=2, dim=-1)

        return out_dict


class HomogeneousGraphSAGE(nn.Module):
    """
    Inductive GraphSAGE encoder for projected homogeneous interaction topology
    with arcsinh outlier compression and residual connections.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.ln2 = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.dropout = dropout

    @staticmethod
    def stabilize_features(x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(torch.clamp(x, -1e6, 1e6))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_stab = self.stabilize_features(x)
        h0 = self.proj(x_stab)
        h1 = h0 + F.gelu(self.ln1(self.conv1(h0, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.res_proj(h0) + self.ln2(self.conv2(h1, edge_index))
        return F.normalize(h2, p=2, dim=-1)
