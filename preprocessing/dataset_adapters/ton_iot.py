"""
Dataset adapter for NF-ToN-IoT-V2 (NetFlow v2 format).
Provides NetFlow flow metrics with L4 port and protocol representations.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, List, Optional
try:
    from preprocessing.label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID
except ImportError:
    from ..label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID


class ToNIoTAdapter:
    """
    Adapter for NF-ToN-IoT-V2 NetFlow parquet dataset.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_sample(self, max_rows: int = 50000) -> Tuple[pd.DataFrame, List[str]]:
        """Loads a uniform sample of NF-ToN-IoT-V2 flows."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ToN-IoT file not found: {self.data_path}")

        t = pq.read_table(self.data_path)
        if max_rows and max_rows < t.num_rows:
            step = t.num_rows // max_rows
            indices = list(range(0, t.num_rows, step))[:max_rows]
            df = t.take(indices).to_pandas()
        else:
            df = t.to_pandas()

        mapped = [map_raw_label(lbl, "TON_IOT") for lbl in df["Attack"]]
        df["is_attack"] = [m[0] for m in mapped]
        df["canonical_family"] = [m[1] for m in mapped]
        df["canonical_id"] = [CANONICAL_TO_ID.get(m[1], CANONICAL_TO_ID["OTHER"]) for m in mapped]
        df["mitre_stage"] = [m[2] for m in mapped]
        df["mitre_id"] = [MITRE_TO_ID.get(m[2], 0) for m in mapped]

        target_cols = ["Label", "Attack", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        feature_cols = [c for c in df.columns if c not in target_cols]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        return df, feature_cols
