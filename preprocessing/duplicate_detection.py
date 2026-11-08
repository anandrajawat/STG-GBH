"""
Duplicate and near-duplicate detection module for STG-GBH.
Ensures zero data leakage between training, validation, and testing partitions.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Set


def compute_row_hashes(df: pd.DataFrame, feature_cols: List[str], precision: int = 4) -> pd.Series:
    """
    Computes MD5 hashes for rows based on specified feature columns.
    Floats are rounded to specified precision to catch near-duplicates.
    """
    sub_df = df[feature_cols].copy()
    for col in sub_df.select_dtypes(include=[np.number]).columns:
        sub_df[col] = sub_df[col].round(precision).fillna(-999.0)
    
    # Serialize row to string and hash
    row_strings = sub_df.astype(str).agg('|'.join, axis=1)
    hashes = row_strings.apply(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest())
    return hashes


def audit_split_duplicates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    Detects intra-split exact duplicates and cross-split duplicate leakage.
    Returns audit statistics dictionary.
    """
    train_hashes = compute_row_hashes(train_df, feature_cols)
    val_hashes = compute_row_hashes(val_df, feature_cols)
    test_hashes = compute_row_hashes(test_df, feature_cols)

    train_set: Set[str] = set(train_hashes)
    val_set: Set[str] = set(val_hashes)
    test_set: Set[str] = set(test_hashes)

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    results = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_unique_hashes": len(train_set),
        "val_unique_hashes": len(val_set),
        "test_unique_hashes": len(test_set),
        "train_internal_duplicate_pct": round((1.0 - len(train_set) / max(len(train_df), 1)) * 100, 2),
        "train_val_leakage_hashes": train_val_overlap,
        "train_test_leakage_hashes": train_test_overlap,
        "val_test_leakage_hashes": val_test_overlap,
        "is_leakage_free": (train_val_overlap == 0 and train_test_overlap == 0)
    }
    return results


def purge_cross_split_leakage(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Removes any record from eval_df that exactly duplicates a training record in feature space.
    """
    train_hashes = set(compute_row_hashes(train_df, feature_cols))
    eval_hashes = compute_row_hashes(eval_df, feature_cols)
    
    mask = ~eval_hashes.isin(train_hashes)
    purged_df = eval_df[mask].copy().reset_index(drop=True)
    return purged_df

