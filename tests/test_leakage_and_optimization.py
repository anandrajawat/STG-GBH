"""
Automated unit tests for threshold optimization, FPR/FNR constraints, and metrics.
"""

import numpy as np
import pytest

from optimization.threshold_optimization import evaluate_threshold, optimize_threshold_constrained
from evaluation.metrics import compute_comprehensive_metrics
from evaluation.calibration import compute_calibration_metrics


def test_threshold_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    res = evaluate_threshold(y_true, y_proba, threshold=0.5)
    assert res["accuracy"] == 1.0
    assert res["fpr"] == 0.0
    assert res["fnr"] == 0.0
    assert np.isclose(res["f1"], 1.0)


def test_constrained_optimization():
    y_true = np.array([0] * 50 + [1] * 50)
    # Well separated with mild noise
    y_proba = np.array([0.05]*48 + [0.3, 0.4] + [0.6, 0.7] + [0.95]*48)

    res = optimize_threshold_constrained(y_true, y_proba, target_max_fpr=0.05, target_max_fnr=0.05)
    assert res["is_constrained_feasible"]
    assert res["status"] == "FEASIBLE_CONSTRAINED"
    assert 0.4 <= res["chosen_threshold"] <= 0.6


def test_calibration_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7], [0.9, 0.1]])

    cal = compute_calibration_metrics(y_true, y_proba)
    assert "expected_calibration_error" in cal
    assert "brier_score" in cal
    assert cal["brier_score"] >= 0.0
