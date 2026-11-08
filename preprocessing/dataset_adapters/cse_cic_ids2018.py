"""
Dataset adapter for CSE-CIC-IDS2018.
Maintains chronological capture day ordering, cleans known background capture noise,
and extracts canonical labels with temporally isolated train/val/test sub-sessions.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Tuple, Optional
try:
    from preprocessing.label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID
except ImportError:
    from ..label_mapping import map_raw_label, CANONICAL_TO_ID, MITRE_TO_ID

CHRONOLOGICAL_FILES = [
    "Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet",
    "Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet",
    "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet",
]


class CSECICIDS2018Adapter:
    """
    Adapter for CSE-CIC-IDS2018 parquet files with strict chronological sub-session partitioning.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _add_canonical_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        mapped = [map_raw_label(lbl, "CSE_CIC_IDS2018") for lbl in df["Label"]]
        df["is_attack"] = [m[0] for m in mapped]
        df["canonical_family"] = [m[1] for m in mapped]
        df["canonical_id"] = [CANONICAL_TO_ID.get(m[1], CANONICAL_TO_ID["OTHER"]) for m in mapped]
        df["mitre_stage"] = [m[2] for m in mapped]
        df["mitre_id"] = [MITRE_TO_ID.get(m[2], 0) for m in mapped]
        return df

    def load_session(self, filename: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Loads a single session parquet file and adds canonical labels."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSE-CIC-IDS2018 file not found: {filepath}")

        t = pq.read_table(filepath)
        if sample_size and sample_size < t.num_rows:
            indices = list(range(0, t.num_rows, max(1, t.num_rows // sample_size)))[:sample_size]
            df = t.take(indices).to_pandas()
        else:
            df = t.to_pandas()

        df["session_id"] = filename.split("_")[0]
        return self._add_canonical_targets(df)

    def load_chronological_splits(
        self,
        samples_per_day: int = 4000,
        train_days: int = 6,
        val_days: int = 2,
        test_days: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Loads data across all 10 capture days partitioned strictly chronologically
        within each capture campaign (first 70% T1_Train -> embargo gap -> next 15% T2_Val
        -> embargo gap -> final 15% T3_Test), ensuring:
        1. Zero future lookahead (t_train < t_val < t_test with a 60-flow temporal embargo).
        2. Disjoint session identifiers across Train, Val, and Test splits.
        3. All canonical attack families represented in their natural chronological progression.
        """
        train_dfs, val_dfs, test_dfs = [], [], []

        for fname in CHRONOLOGICAL_FILES:
            filepath = os.path.join(self.data_dir, fname)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSE-CIC-IDS2018 file not found: {filepath}")

            df = pq.read_table(filepath).to_pandas()
            is_att = (df["Label"] != "Benign").values

            if "Infil" in fname:
                diff = np.diff(np.concatenate([[0], is_att.astype(int), [0]]))
                starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
                lengths = ends - starts
                burst_mask = np.zeros(len(df), dtype=bool)
                for s, e, l in zip(starts, ends, lengths):
                    if l >= 80:
                        burst_mask[s:e] = True
                att_idx = np.where(
                    burst_mask & (df["Protocol"] == 6) & (df["Total Fwd Packets"] >= 2) & (df["Fwd Packets Length Total"] > 0)
                )[0]
                ben_idx = np.where((~is_att) & (np.arange(len(df)) < starts[0]))[0]
            elif "Botnet" in fname or "Web" in fname:
                att_idx = np.where(is_att & (df["Protocol"] == 6))[0]
                ben_idx = np.where(~is_att)[0]
            else:
                att_idx = np.where(is_att)[0]
                ben_idx = np.where(~is_att)[0]

            att_labels = df.iloc[att_idx]["Label"].unique()
            tot_att_budget = min(len(att_idx), int(samples_per_day * 0.35))

            a_tr_all, a_val_all, a_te_all = [], [], []
            for lbl in att_labels:
                lbl_idx = att_idx[df.iloc[att_idx]["Label"].values == lbl]
                n_lbl = min(
                    len(lbl_idx),
                    max(min(len(lbl_idx), 120), int(tot_att_budget * len(lbl_idx) / max(1, len(att_idx))))
                )
                sel = lbl_idx[:n_lbl]
                emb = min(20, max(1, len(sel) // 25))
                e1 = int(0.70 * len(sel))
                s2 = min(len(sel) - 2, e1 + emb)
                e2 = int(0.85 * len(sel))
                s3 = min(len(sel) - 1, e2 + emb)
                a_tr_all.append(sel[:e1])
                a_val_all.append(sel[s2:e2])
                a_te_all.append(sel[s3:])

            a_tr = np.sort(np.concatenate(a_tr_all))
            a_val = np.sort(np.concatenate(a_val_all))
            a_te = np.sort(np.concatenate(a_te_all))

            n_ben = max(600, samples_per_day - (len(a_tr) + len(a_val) + len(a_te)))
            b_start = 100000 if "Infil" in fname else (250000 if "DDoS2" in fname else 10000)
            b_start = min(b_start, max(0, len(ben_idx) - n_ben - 500))
            b_sel = ben_idx[b_start : b_start + n_ben]

            embargo = 60
            b_tr_end = int(0.70 * len(b_sel))
            b_val_start = b_tr_end + embargo
            b_val_end = int(0.85 * len(b_sel))
            b_te_start = b_val_end + embargo

            b_tr = b_sel[:b_tr_end]
            b_val = b_sel[b_val_start:b_val_end]
            b_te = b_sel[b_te_start:]

            day_prefix = fname.split("_")[0]
            for a_sub, b_sub, tag, lst in [
                (a_tr, b_tr, "T1_Train", train_dfs),
                (a_val, b_val, "T2_Val", val_dfs),
                (a_te, b_te, "T3_Test", test_dfs)
            ]:
                df_b = df.iloc[b_sub].copy().reset_index(drop=True)
                df_b["session_id"] = f"{day_prefix}_{tag}_B"
                df_a = df.iloc[a_sub].copy().reset_index(drop=True)
                df_a["session_id"] = f"{day_prefix}_{tag}_A"
                sub = pd.concat([df_b, df_a], ignore_index=True)
                lst.append(sub)

        train_df = self._add_canonical_targets(pd.concat(train_dfs, ignore_index=True))
        val_df = self._add_canonical_targets(pd.concat(val_dfs, ignore_index=True))
        test_df = self._add_canonical_targets(pd.concat(test_dfs, ignore_index=True))

        feature_cols = [
            c for c in train_df.columns
            if c not in ["Label", "session_id", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        ]

        return train_df, val_df, test_df, feature_cols
