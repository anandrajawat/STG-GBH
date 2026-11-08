"""
Dataset adapter for CIC-IDS2017.
Enables cross-dataset evaluation against CSE-CIC-IDS2018 with identical feature schema.
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Tuple, Optional
try:
    from preprocessing.label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID
except ImportError:
    from ..label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID

FILES_2017 = [
    "Benign-Monday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet"
]


class CICIDS2017Adapter:
    """
    Adapter for CIC-IDS2017 cross-dataset evaluation.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_all(self, samples_per_file: int = 15000) -> Tuple[pd.DataFrame, List[str]]:
        """Loads and prepares CIC-IDS2017 for cross-dataset zero-shot evaluation."""
        dfs = []
        for fname in FILES_2017:
            filepath = os.path.join(self.data_dir, fname)
            if not os.path.exists(filepath):
                continue
            t = pq.read_table(filepath)
            if samples_per_file and samples_per_file < t.num_rows:
                indices = list(range(0, t.num_rows, max(1, t.num_rows // samples_per_file)))[:samples_per_file]
                df = t.take(indices).to_pandas()
            else:
                df = t.to_pandas()

            df["source_dataset"] = "CIC_IDS2017"
            df["session_file"] = fname
            dfs.append(df)

        comb_df = pd.concat(dfs, ignore_index=True)
        mapped = [map_raw_label(lbl, "CIC_IDS2017") for lbl in comb_df["Label"]]
        comb_df["is_attack"] = [m[0] for m in mapped]
        comb_df["canonical_family"] = [m[1] for m in mapped]
        comb_df["canonical_id"] = [CANONICAL_TO_ID.get(m[1], CANONICAL_TO_ID["OTHER"]) for m in mapped]
        comb_df["mitre_stage"] = [m[2] for m in mapped]
        comb_df["mitre_id"] = [MITRE_TO_ID.get(m[2], 0) for m in mapped]

        feature_cols = [
            c for c in comb_df.columns 
            if c not in ["Label", "source_dataset", "session_file", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        ]
        return comb_df, feature_cols
