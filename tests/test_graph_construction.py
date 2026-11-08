"""
Automated unit tests for heterogeneous graph construction and temporal snapshots.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from graph_construction.entity_extraction import EntityRegistry, extract_entities
from graph_construction.relation_extraction import build_hetero_graph
from graph_construction.graph_snapshots import TemporalSnapshotManager
from graph_construction.temporal_graph import TemporalGraphPipeline, GraphTopologyUnavailableError


def test_hetero_graph_construction():
    registry = EntityRegistry()
    df = pd.DataFrame({
        "ip.src_host": ["192.168.1.1", "192.168.1.2"],
        "ip.dst_host": ["10.0.0.1", "10.0.0.2"],
        "tcp.dstport": [80, 443],
        "tcp.flags": [2, 18],
        "is_attack": [0, 1]
    })
    feats = np.random.randn(2, 8).astype(np.float32)

    g = build_hetero_graph(df, registry, feats, df["is_attack"].values)
    assert "flow" in g.node_types
    assert "src_ip" in g.node_types
    assert g["flow"].x.shape == (2, 8)
    assert g["flow"].y.shape == (2,)
    assert ("src_ip", "COMMUNICATES_WITH", "dst_ip") in g.edge_types


def test_temporal_snapshots_integrity():
    mgr = TemporalSnapshotManager(window_size_str="15m")
    df = pd.DataFrame({
        "ip.src_host": ["192.168.1.1", "192.168.1.2", "192.168.1.1"],
        "ip.dst_host": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "tcp.dstport": [80, 443, 80],
        "tcp.flags": [2, 18, 2],
        "timestamp_sec": [100.0, 500.0, 1200.0],
        "is_attack": [0, 1, 0]
    })
    feats = np.random.randn(3, 8).astype(np.float32)
    snapshots = mgr.create_snapshots(df, feats)
    assert len(snapshots) >= 1
    for t_start, g_t in snapshots:
        mgr.verify_snapshot_integrity(g_t)


def test_graph_topology_unavailable_guard():
    pipeline = TemporalGraphPipeline()
    # Missing IP columns should raise GraphTopologyUnavailableError rather than inventing fake graph
    df_stripped = pd.DataFrame({"Flow_Duration": [100, 200]})
    with pytest.raises(GraphTopologyUnavailableError):
        pipeline.build_snapshots_from_df(df_stripped, np.zeros((2, 1)))

