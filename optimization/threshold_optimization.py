"""
FPR and FNR Optimization and Cost-Sensitive Threshold Analysis for STG-GBH.
Calculates constrained thresholds (FPR <= 2%, FNR <= 2%), alerts per 10k flows,
and cost-sensitive operating points.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score, average_precision_score
)


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0
) -> Dict[str, float]:
    """
    Computes all operational detection metrics at a specific classification threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / float(fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / float(fn + tp) if (fn + tp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / float(len(y_true)) if len(y_true) > 0 else 0.0

    # F1 and Macro-F1
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    benign_prec = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
    benign_rec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    benign_f1 = 2 * benign_prec * benign_rec / (benign_prec + benign_rec + 1e-12)
    macro_f1 = 0.5 * (f1 + benign_f1)

    # Operational Security Metrics
    fp_per_10k_benign = fpr * 10000.0
    fn_per_10k_malicious = fnr * 10000.0
    operational_cost = (fp * cost_fp + fn * cost_fn) / float(len(y_true))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fp_per_10k_benign": float(fp_per_10k_benign),
        "fn_per_10k_malicious": float(fn_per_10k_malicious),
        "operational_cost": float(operational_cost)
    }


def optimize_threshold_constrained(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_max_fpr: float = 0.02,
    target_max_fnr: float = 0.02,
    num_thresholds: int = 200,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0
) -> Dict[str, Any]:
    """
    Sweeps thresholds and identifies:
      1. Constrained Optimum: Maximize Macro-F1 subject to FPR <= target_max_fpr and FNR <= target_max_fnr.
      2. If infeasible, Pareto Best: Maximizes Macro-F1 while minimizing (FPR + FNR).
      3. Cost-Optimal threshold: Minimizes cost-weighted operational loss.
    """
    threshold_grid = np.linspace(0.15, 0.85, num_thresholds)
    all_evals: List[Dict[str, float]] = []

    feasible_candidates = []
    best_pareto = None
    best_pareto_score = -float("inf")
    best_cost = None
    min_cost = float("inf")

    for th in threshold_grid:
        res = evaluate_threshold(y_true, y_proba, th, cost_fp=cost_fp, cost_fn=cost_fn)
        all_evals.append(res)

        if res["fpr"] <= target_max_fpr and res["fnr"] <= target_max_fnr:
            feasible_candidates.append(res)

        # Pareto objective: high Macro-F1 and low (FPR + FNR)
        pareto_score = res["macro_f1"] - 0.5 * (res["fpr"] + res["fnr"])
        if pareto_score > best_pareto_score:
            best_pareto_score = pareto_score
            best_pareto = res

        if res["operational_cost"] < min_cost:
            min_cost = res["operational_cost"]
            best_cost = res

    if len(feasible_candidates) > 0:
        feasible_candidates.sort(
            key=lambda x: x["macro_f1"] - 0.15 * abs(x["threshold"] - 0.50),
            reverse=True
        )
        chosen_point = feasible_candidates[0]
        status = "FEASIBLE_CONSTRAINED"
    else:
        chosen_point = best_pareto
        status = "INFEASIBLE_PARETO_FALLBACK"

    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "status": status,
        "chosen_threshold": chosen_point["threshold"],
        "chosen_metrics": chosen_point,
        "is_constrained_feasible": (len(feasible_candidates) > 0),
        "target_max_fpr": target_max_fpr,
        "target_max_fnr": target_max_fnr,
        "best_cost_threshold": best_cost["threshold"],
        "best_cost_metrics": best_cost,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "all_evaluations": all_evals
    }
