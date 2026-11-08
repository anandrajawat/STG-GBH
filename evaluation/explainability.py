"""
Model Explainability and Interpretability Module for STG-GBH.
Computes Tree SHAP values, feature importance rankings, and attention visualizations.
"""

import os
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def compute_tree_shap_summary(
    tree_model,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_display: int = 15
) -> Dict[str, Any]:
    """
    Computes Tree SHAP values for tree-based models and extracts top features.
    """
    # Subsample if too large for fast computation
    if len(X_sample) > 500:
        indices = np.random.RandomState(42).choice(len(X_sample), 500, replace=False)
        sub_X = X_sample[indices]
    else:
        sub_X = X_sample

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(sub_X)

    # Handle multi-class vs binary shap values
    if isinstance(shap_values, list):
        # Multiclass or binary list
        vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    elif shap_values.ndim == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]

    top_features = [
        {
            "rank": int(r + 1),
            "feature": feature_names[idx] if idx < len(feature_names) else f"feat_{idx}",
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 5)
        }
        for r, idx in enumerate(top_indices)
    ]

    return {
        "top_features": top_features,
        "mean_abs_shap": mean_abs_shap.tolist(),
        "sample_size": len(sub_X)
    }


def compare_feature_importances(
    xgb_importances: np.ndarray,
    cat_importances: np.ndarray,
    lgb_importances: np.ndarray,
    feature_names: List[str],
    top_k: int = 15
) -> pd.DataFrame:
    """
    Constructs normalized comparative importance table across all three boosting architectures.
    """
    def _norm(arr):
        s = np.sum(arr)
        return (arr / s) if s > 0 else arr

    n_feats = min(len(feature_names), len(xgb_importances), len(cat_importances), len(lgb_importances))
    names = feature_names[:n_feats]
    
    norm_xgb = _norm(xgb_importances[:n_feats])
    norm_cat = _norm(cat_importances[:n_feats])
    norm_lgb = _norm(lgb_importances[:n_feats])
    mean_imp = (norm_xgb + norm_cat + norm_lgb) / 3.0

    df = pd.DataFrame({
        "Feature": names,
        "XGBoost_Norm_Importance": norm_xgb,
        "CatBoost_Norm_Importance": norm_cat,
        "LightGBM_Norm_Importance": norm_lgb,
        "Ensemble_Mean_Importance": mean_imp
    }).sort_values(by="Ensemble_Mean_Importance", ascending=False).head(top_k).reset_index(drop=True)

    return df

