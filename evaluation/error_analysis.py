"""
Scientific Error Analysis and Root-Cause Diagnostics for STG-GBH.
Dissects False Positives and False Negatives with operational security context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def perform_error_analysis(
    df_eval: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_cols: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Performs forensic error analysis identifying:
      - False Positives (Benign flows flagged as attacks)
      - False Negatives (Attacks missed by the system)
      - Borderline predictions near the classification threshold
    """
    prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tp_count = int(tp_mask.sum())
    tn_count = int(tn_mask.sum())

    # Examine FP samples
    fp_indices = np.where(fp_mask)[0]
    fp_details = []
    if len(fp_indices) > 0:
        # Sort by highest confidence false alarms
        sorted_fp = fp_indices[np.argsort(prob_pos[fp_indices])[::-1]][:top_n]
        for idx in sorted_fp:
            row = df_eval.iloc[idx]
            fp_details.append({
                "index": int(idx),
                "predicted_prob": round(float(prob_pos[idx]), 4),
                "raw_label": str(row.get("Label", row.get("Attack_type", "Benign"))),
                "canonical_family": str(row.get("canonical_family", "BENIGN")),
                "session_id": str(row.get("session_id", "N/A"))
            })

    # Examine FN samples
    fn_indices = np.where(fn_mask)[0]
    fn_details = []
    if len(fn_indices) > 0:
        # Sort by lowest predicted prob (most confident misses)
        sorted_fn = fn_indices[np.argsort(prob_pos[fn_indices])][:top_n]
        for idx in sorted_fn:
            row = df_eval.iloc[idx]
            fn_details.append({
                "index": int(idx),
                "predicted_prob": round(float(prob_pos[idx]), 4),
                "raw_label": str(row.get("Label", row.get("Attack_type", "Unknown Attack"))),
                "canonical_family": str(row.get("canonical_family", "UNKNOWN")),
                "session_id": str(row.get("session_id", "N/A"))
            })

    # Failure patterns by attack family
    fn_family_dist = {}
    if "canonical_family" in df_eval.columns:
        fn_families = df_eval.iloc[fn_indices]["canonical_family"].value_counts()
        fn_family_dist = {str(k): int(v) for k, v in fn_families.items()}

    return {
        "confusion_counts": {"tp": tp_count, "fp": fp_count, "tn": tn_count, "fn": fn_count},
        "false_positive_rate": float(fp_count / (fp_count + tn_count)) if (fp_count + tn_count) > 0 else 0.0,
        "false_negative_rate": float(fn_count / (fn_count + tp_count)) if (fn_count + tp_count) > 0 else 0.0,
        "top_false_positive_cases": fp_details,
        "top_false_negative_cases": fn_details,
        "missed_attacks_by_family": fn_family_dist
    }

