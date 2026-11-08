"""
Dataset adapter for CICIoT2023 large-scale IoT intrusion dataset.
Supports chunked reading across partitioned CSV files.
"""

import os
import glob
import pandas as pd
from typing import Tuple, List, Optional
try:
    from preprocessing.label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID
except ImportError:
    from ..label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID


class CICIoT2023Adapter:
    """
    Adapter for CICIoT2023 partition files.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_sample(self, max_rows: int = 50000) -> Tuple[pd.DataFrame, List[str]]:
        """Loads a representative sample across available CSV partitions."""
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.csv"), recursive=True))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        per_file = max(1000, max_rows // len(csv_files))
        dfs = []
        for f in csv_files:
            df = pd.read_csv(f, nrows=per_file)
            dfs.append(df)
            if sum(len(x) for x in dfs) >= max_rows:
                break

        comb_df = pd.concat(dfs, ignore_index=True)
        if len(comb_df) > max_rows:
            comb_df = comb_df.iloc[:max_rows].copy()

        # Map labels
        mapped = [map_raw_label(lbl, "CICIOT2023") for lbl in comb_df["label"]]
        comb_df["is_attack"] = [m[0] for m in mapped]
        comb_df["canonical_family"] = [m[1] for m in mapped]
        comb_df["canonical_id"] = [CANONICAL_TO_ID.get(m[1], CANONICAL_TO_ID["OTHER"]) for m in mapped]
        comb_df["mitre_stage"] = [m[2] for m in mapped]
        comb_df["mitre_id"] = [MITRE_TO_ID.get(m[2], 0) for m in mapped]

        target_cols = ["label", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        feature_cols = [c for c in comb_df.columns if c not in target_cols]
        for c in feature_cols:
            comb_df[c] = pd.to_numeric(comb_df[c], errors="coerce").fillna(0.0)

        return comb_df, feature_cols
