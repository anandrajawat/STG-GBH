"""
Relation and edge extraction for heterogeneous cybersecurity graphs in STG-GBH.
Builds PyG HeteroData objects representing multi-relational network topology.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from torch_geometric.data import HeteroData
from .entity_extraction import EntityRegistry, extract_entities


def build_hetero_graph(
    df: pd.DataFrame,
    registry: EntityRegistry,
    feature_matrix: np.ndarray,
    target_labels: Optional[np.ndarray] = None
) -> HeteroData:
    """
    Constructs a PyG HeteroData object from flow dataframe and feature matrix.
    Edges:
      - (src_ip, COMMUNICATES_WITH, dst_ip)
      - (src_ip, INITIATES_FLOW, flow)
      - (flow, TARGETS, dst_ip)
      - (flow, USES_SERVICE, service)
      - (flow, USES_PROTOCOL, protocol)
    """
    entities = extract_entities(df, registry)
    n_flows = len(df)
    data = HeteroData()

    # Flow node features
    data["flow"].x = torch.tensor(feature_matrix, dtype=torch.float32)
    if target_labels is not None:
        data["flow"].y = torch.tensor(target_labels, dtype=torch.long)

    # Initial node representations for IPs, services, protocols
    num_src = registry.get_count("src_ip")
    num_dst = registry.get_count("dst_ip")
    num_srv = registry.get_count("service")
    num_proto = registry.get_count("protocol")

    # Feature dimension matches flow feature dim via embedding/projection or degree
    feat_dim = feature_matrix.shape[1]
    data["src_ip"].x = torch.zeros((num_src, feat_dim), dtype=torch.float32)
    data["dst_ip"].x = torch.zeros((num_dst, feat_dim), dtype=torch.float32)
    data["service"].x = torch.zeros((num_srv, feat_dim), dtype=torch.float32)
    data["protocol"].x = torch.zeros((num_proto, feat_dim), dtype=torch.float32)

    # Populate basic degree features into node features
    src_deg = np.bincount(entities["src_ip"], minlength=num_src)
    dst_deg = np.bincount(entities["dst_ip"], minlength=num_dst)
    data["src_ip"].x[:, 0] = torch.tensor(np.log1p(src_deg), dtype=torch.float32)
    data["dst_ip"].x[:, 0] = torch.tensor(np.log1p(dst_deg), dtype=torch.float32)

    # Edges
    src_t = torch.tensor(entities["src_ip"], dtype=torch.long)
    dst_t = torch.tensor(entities["dst_ip"], dtype=torch.long)
    flow_t = torch.tensor(range(n_flows), dtype=torch.long)
    srv_t = torch.tensor(entities["service"], dtype=torch.long)
    proto_t = torch.tensor(entities["protocol"], dtype=torch.long)

    data["src_ip", "COMMUNICATES_WITH", "dst_ip"].edge_index = torch.stack([src_t, dst_t], dim=0)
    data["src_ip", "INITIATES_FLOW", "flow"].edge_index = torch.stack([src_t, flow_t], dim=0)
    data["flow", "TARGETS", "dst_ip"].edge_index = torch.stack([flow_t, dst_t], dim=0)
    data["flow", "USES_SERVICE", "service"].edge_index = torch.stack([flow_t, srv_t], dim=0)
    data["flow", "USES_PROTOCOL", "protocol"].edge_index = torch.stack([flow_t, proto_t], dim=0)

    return data

