"""
Temporal graph snapshot generator for STG-GBH.
Discretizes streaming flows into dynamic temporal snapshots G_t = (V_t, E_t, X_t).
Supports 5m, 15m, 30m, and 60m windows.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import HeteroData
from .entity_extraction import EntityRegistry
from .relation_extraction import build_hetero_graph


class TemporalSnapshotManager:
    """
    Partitions temporal events into time-windowed heterogeneous graph snapshots.
    """
    WINDOW_MINUTES = {
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "60m": 60 * 60
    }

    def __init__(self, window_size_str: str = "60m"):
        if window_size_str not in self.WINDOW_MINUTES:
            raise ValueError(f"Unsupported window size '{window_size_str}'. Choose from {list(self.WINDOW_MINUTES.keys())}")
        self.window_size_str = window_size_str
        self.window_sec = self.WINDOW_MINUTES[window_size_str]
        self.registry = EntityRegistry()

    def create_snapshots(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
        time_col: str = "timestamp_sec",
        target_col: str = "is_attack"
    ) -> List[Tuple[float, HeteroData]]:
        """
        Generates sequence of temporal snapshots [(t_start, HeteroData), ...].
        """
        if time_col not in df.columns:
            raise KeyError(f"Timestamp column '{time_col}' required for dynamic temporal snapshots!")

        times = df[time_col].values
        t_min = float(times.min())
        t_max = float(times.max())

        snapshots = []
        cur_t = t_min

        while cur_t <= t_max:
            next_t = cur_t + self.window_sec
            mask = (times >= cur_t) & (times < next_t)
            sub_df = df[mask]
            
            if len(sub_df) > 0:
                sub_feats = feature_matrix[mask]
                sub_targets = df[target_col].values[mask] if target_col in df.columns else None
                
                # Build snapshot
                g_t = build_hetero_graph(sub_df, self.registry, sub_feats, sub_targets)
                self.verify_snapshot_integrity(g_t)
                snapshots.append((cur_t, g_t))

            cur_t = next_t

        return snapshots

    @staticmethod
    def verify_snapshot_integrity(g: HeteroData):
        """Validates that graph tensors are contiguous, non-empty, and indices are in bounds."""
        assert "flow" in g.node_types, "Graph missing 'flow' node type!"
        assert g["flow"].x is not None, "Flow nodes missing feature tensor x!"
        assert not torch.isnan(g["flow"].x).any(), "NaN detected in flow node features!"
        assert not torch.isinf(g["flow"].x).any(), "Inf detected in flow node features!"

        # Check edge index ranges
        for edge_type in g.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = g[edge_type].edge_index
            if edge_index.shape[1] > 0:
                max_src = edge_index[0].max().item()
                max_dst = edge_index[1].max().item()
                num_src = g[src_type].num_nodes
                num_dst = g[dst_type].num_nodes
                assert max_src < num_src, f"Edge src index {max_src} out of bounds for {src_type} ({num_src})!"
                assert max_dst < num_dst, f"Edge dst index {max_dst} out of bounds for {dst_type} ({num_dst})!"

