"""
Probability calibration assessment and Expected Calibration Error (ECE) for STG-GBH.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculates ECE, MCE, and reliability curve coordinates.
    """
    prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    prob_true, prob_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins, strategy="uniform")

    # Compute ECE (Expected Calibration Error)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prob_pos, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = mask.sum()
        if count > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = prob_pos[mask].mean()
            diff = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * diff
            mce = max(mce, diff)

    brier = float(brier_score_loss(y_true, prob_pos))

    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": brier,
        "prob_true": [float(x) for x in prob_true],
        "prob_pred": [float(x) for x in prob_pred]
    }

