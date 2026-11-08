"""
Temporal graph orchestrator and neighborhood sampling for STG-GBH.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from .graph_snapshots import TemporalSnapshotManager


class GraphTopologyUnavailableError(Exception):
    """Raised when a dataset lacks sufficient IP/temporal entities for graph construction."""
    pass


class TemporalGraphPipeline:
    """
    Coordinates dynamic graph snapshot generation and neighbor sampling.
    """
    def __init__(self, window_size: str = "60m"):
        self.window_size = window_size
        self.snapshot_mgr = TemporalSnapshotManager(window_size)
        self.snapshots: List[Tuple[float, HeteroData]] = []

    def build_snapshots_from_df(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
        time_col: str = "timestamp_sec",
        target_col: str = "is_attack"
    ) -> List[Tuple[float, HeteroData]]:
        """
        Builds dynamic snapshots or raises GraphTopologyUnavailableError if entity info is missing.
        """
        if "ip.src_host" not in df.columns or "ip.dst_host" not in df.columns:
            raise GraphTopologyUnavailableError(
                "Dataset lacks genuine IP endpoint identifiers. In accordance with Part 5 research rules, "
                "graph topology experiments are marked UNAVAILABLE rather than inventing artificial topology."
            )

        self.snapshots = self.snapshot_mgr.create_snapshots(
            df, feature_matrix, time_col=time_col, target_col=target_col
        )
        return self.snapshots

    def get_flow_neighbor_loader(
        self,
        snapshot_idx: int,
        batch_size: int = 256,
        num_neighbors: Dict[Tuple[str, str, str], List[int]] = None
    ) -> NeighborLoader:
        """
        Returns an inductive PyG NeighborLoader for flow nodes in a given snapshot.
        """
        if snapshot_idx >= len(self.snapshots):
            raise IndexError(f"Snapshot index {snapshot_idx} out of range ({len(self.snapshots)} available).")

        _, g = self.snapshots[snapshot_idx]
        if num_neighbors is None:
            # Default 2-layer neighbor sampling: [15, 10]
            num_neighbors = {edge_type: [15, 10] for edge_type in g.edge_types}

        loader = NeighborLoader(
            g,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=("flow", torch.arange(g["flow"].num_nodes)),
            shuffle=False
        )
        return loader

