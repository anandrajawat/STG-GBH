"""
Dataset adapter for Edge-IIoTset.
Handles timestamp parsing, IP/Port validation, corruption filtering, and canonical mapping.
Provides entity graphs and tabular ML features.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
try:
    from preprocessing.label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID
except ImportError:
    from ..label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID


class EdgeIIoTsetAdapter:
    """
    Adapter for Edge-IIoTset benchmark dataset.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_clean_data(self, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Loads, purges corrupted records (shifted CSV rows), parses timestamps,
        and generates canonical targets.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Edge-IIoTset file not found: {self.data_path}")

        df = pd.read_csv(self.data_path, low_memory=False)

        # 1. Filter corrupted rows lacking valid 2021 timestamp string
        valid_ts_mask = df['frame.time'].astype(str).str.contains('2021', na=False)
        clean_df = df[valid_ts_mask].copy().reset_index(drop=True)

        if max_rows and max_rows < len(clean_df):
            # Take evenly spaced samples across the timeline
            step = len(clean_df) // max_rows
            clean_df = clean_df.iloc[::step].iloc[:max_rows].copy().reset_index(drop=True)

        # 2. Parse timestamps to datetime and POSIX seconds
        # Timestamps look like: ' 2021 23:24:32.646740000 '
        clean_df['datetime'] = pd.to_datetime(clean_df['frame.time'].str.strip(), format='%Y %H:%M:%S.%f', errors='coerce')
        # Fill any unparsed timestamps using forward/backward fill along the chronological capture
        clean_df['datetime'] = clean_df['datetime'].ffill().bfill()
        clean_df['timestamp_sec'] = clean_df['datetime'].astype('int64') // 10**9

        # 3. Canonical targets
        mapped = [map_raw_label(lbl, "EDGE_IIOTSET") for lbl in clean_df["Attack_type"]]
        clean_df["is_attack"] = [m[0] for m in mapped]
        clean_df["canonical_family"] = [m[1] for m in mapped]
        clean_df["canonical_id"] = [CANONICAL_TO_ID.get(m[1], CANONICAL_TO_ID["OTHER"]) for m in mapped]
        clean_df["mitre_stage"] = [m[2] for m in mapped]
        clean_df["mitre_id"] = [MITRE_TO_ID.get(m[2], 0) for m in mapped]

        # 4. Separate graph entity columns from numeric predictive features
        entity_cols = [
            'frame.time', 'datetime', 'timestamp_sec',
            'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
            'tcp.srcport', 'tcp.dstport', 'udp.port'
        ]
        target_cols = [
            'Attack_label', 'Attack_type', 'is_attack', 'canonical_family', 'canonical_id', 'mitre_stage', 'mitre_id'
        ]

        # Convert remaining features to numeric
        feature_cols = []
        for col in clean_df.columns:
            if col not in entity_cols and col not in target_cols:
                # Attempt conversion to numeric
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)
                feature_cols.append(col)

        return clean_df, feature_cols
