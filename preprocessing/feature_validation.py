"""
Feature validation and schema integrity checking for STG-GBH.
Verifies no target-derived features or infinite values compromise the pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


class FeatureValidator:
    """
    Validates features for numerical integrity, target leakage, and distribution shifts.
    """
    FORBIDDEN_COL_SUBSTRINGS = [
        "label", "attack", "mitre", "ground_truth", "target", "class", "stage", "category"
    ]

    def __init__(self, allowed_features: List[str]):
        self.allowed_features = list(allowed_features)
        self._check_for_target_leakage(self.allowed_features)

    def _check_for_target_leakage(self, feature_names: List[str]):
        """Ensures no feature name hints at ground truth labels or targets."""
        for feat in feature_names:
            norm = feat.lower()
            if any(forbidden in norm for forbidden in self.FORBIDDEN_COL_SUBSTRINGS):
                # Exception: legitimate network protocol flag names like 'ack_flag_number'
                if norm in ["ack_flag_number", "fin_flag_number", "syn_flag_number", "rst_flag_number", 
                            "psh_flag_number", "ece_flag_number", "cwr_flag_number", "attack_independent_deg"]:
                    continue
                raise ValueError(f"CRITICAL LEAKAGE HAZARD: Feature '{feat}' matches forbidden substring!")

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data integrity checks on input DataFrame.
        """
        missing_cols = [c for c in self.allowed_features if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required feature columns: {missing_cols[:5]}...")

        sub_df = df[self.allowed_features]
        num_df = sub_df.select_dtypes(include=[np.number])
        
        nan_counts = int(num_df.isna().sum().sum())
        pos_inf = int(np.isposinf(num_df.values).sum())
        neg_inf = int(np.isneginf(num_df.values).sum())
        
        # Check for constant/zero-variance features
        std_vals = num_df.std()
        zero_var_cols = list(std_vals[std_vals == 0].index)

        return {
            "num_rows": len(df),
            "num_features": len(self.allowed_features),
            "nan_count": nan_counts,
            "pos_inf_count": pos_inf,
            "neg_inf_count": neg_inf,
            "zero_variance_cols": zero_var_cols,
            "is_clean": (nan_counts == 0 and pos_inf == 0 and neg_inf == 0)
        }


def clean_infinities_and_nans(df: pd.DataFrame, feature_cols: List[str], fill_medians: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Replaces infs and NaNs strictly using medians computed on training set.
    """
    clean_df = df.copy()
    if fill_medians is None:
        fill_medians = {}
        for col in feature_cols:
            if col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[col]):
                series = clean_df[col].replace([np.inf, -np.inf], np.nan)
                med = float(series.median())
                if np.isnan(med):
                    med = 0.0
                fill_medians[col] = med

    for col in feature_cols:
        if col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].replace([np.inf, -np.inf], np.nan).fillna(fill_medians[col])

    return clean_df, fill_medians

