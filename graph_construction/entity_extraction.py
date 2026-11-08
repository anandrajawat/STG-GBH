"""
Entity extraction for heterogeneous cybersecurity graphs in STG-GBH.
Extracts distinct typed nodes: src_ip, dst_ip, protocol, service, flow.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any


class EntityRegistry:
    """
    Maintains typed entity indices across graph snapshots.
    """
    def __init__(self):
        self.node_to_id: Dict[str, Dict[str, int]] = {
            "src_ip": {},
            "dst_ip": {},
            "protocol": {},
            "service": {},
            "flow": {}
        }

    def get_or_create(self, node_type: str, raw_val: Any) -> int:
        val_str = str(raw_val).strip()
        reg = self.node_to_id[node_type]
        if val_str not in reg:
            reg[val_str] = len(reg)
        return reg[val_str]

    def get_count(self, node_type: str) -> int:
        return len(self.node_to_id[node_type])


def extract_entities(
    df: pd.DataFrame,
    registry: EntityRegistry,
    src_ip_col: str = "ip.src_host",
    dst_ip_col: str = "ip.dst_host",
    port_col: str = "tcp.dstport",
    proto_col: str = "tcp.flags"
) -> Dict[str, np.ndarray]:
    """
    Extracts entity IDs for every record in the dataframe.
    """
    n = len(df)
    src_ids = np.zeros(n, dtype=np.int64)
    dst_ids = np.zeros(n, dtype=np.int64)
    proto_ids = np.zeros(n, dtype=np.int64)
    service_ids = np.zeros(n, dtype=np.int64)
    flow_ids = np.zeros(n, dtype=np.int64)

    has_src = src_ip_col in df.columns
    has_dst = dst_ip_col in df.columns
    has_port = port_col in df.columns
    has_proto = proto_col in df.columns

    for i in range(n):
        s_ip = df.iloc[i][src_ip_col] if has_src else f"src_{i}"
        d_ip = df.iloc[i][dst_ip_col] if has_dst else f"dst_{i}"
        port = df.iloc[i][port_col] if has_port else "0"
        proto = df.iloc[i][proto_col] if has_proto else "default_proto"

        src_ids[i] = registry.get_or_create("src_ip", s_ip)
        dst_ids[i] = registry.get_or_create("dst_ip", d_ip)
        service_ids[i] = registry.get_or_create("service", port)
        proto_ids[i] = registry.get_or_create("protocol", proto)
        flow_ids[i] = registry.get_or_create("flow", f"flow_{df.index[i]}")

    return {
        "src_ip": src_ids,
        "dst_ip": dst_ids,
        "service": service_ids,
        "protocol": proto_ids,
        "flow": flow_ids
    }

