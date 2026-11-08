"""
Statistical Validation and Hypothesis Testing for STG-GBH.
Calculates 95% Confidence Intervals, Paired t-tests, McNemar tests,
and Cohen's d effect sizes across 5 independent random seeds.
"""

import os
import json
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def compute_mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """
    Computes McNemar's paired test for binary classification models.
    Contingency table:
      n00: both correct
      n01: A correct, B wrong
      n10: A wrong, B correct
      n11: both wrong
    Chi-squared statistic with continuity correction: (|n01 - n10| - 1)^2 / (n01 + n10)
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b)) # A right, B wrong
    c = int(np.sum(~correct_a & correct_b)) # A wrong, B right

    denom = b + c
    if denom == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_05": False}

    chi2 = (abs(b - c) - 1.0)**2 / float(denom)
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "b_model_a_better": b,
        "c_model_b_better": c,
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(p_val, 6),
        "significant_at_05": (p_val < 0.05)
    }


def compute_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Computes mean, standard deviation, and Student's t 95% confidence interval."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    else:
        h = 0.0

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(mean - h, 4),
        "ci_upper": round(mean + h, 4)
    }


def compute_cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Computes Cohen's d effect size between two metric distributions."""
    a, b = np.array(group_a), np.array(group_b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1))
    if pooled_sd == 0:
        return 0.0
    d = float((np.mean(a) - np.mean(b)) / pooled_sd)
    return round(d, 4)


class StatisticalValidationRunner:
    """
    Runs multi-seed validation and generates rigorous statistical comparisons.
    """
    def __init__(self, output_dir: str = "results", n_seeds: int = 5):
        self.output_dir = output_dir
        self.n_seeds = n_seeds
        self.seeds = [42 + i for i in range(n_seeds)]

    def run_multi_seed_comparison(
        self,
        baseline_metrics_dict: Dict[str, List[float]],
        proposed_metrics: List[float]
    ) -> Dict[str, Any]:
        report = {
            "proposed_model_ci": compute_confidence_intervals(proposed_metrics),
            "pairwise_comparisons": {}
        }

        for model_name, base_vals in baseline_metrics_dict.items():
            base_ci = compute_confidence_intervals(base_vals)
            t_stat, p_val = stats.ttest_rel(proposed_metrics, base_vals) if len(base_vals) == len(proposed_metrics) else (0.0, 1.0)
            d = compute_cohens_d(proposed_metrics, base_vals)

            report["pairwise_comparisons"][model_name] = {
                "baseline_ci": base_ci,
                "paired_t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_at_05": bool(p_val < 0.05),
                "cohens_d_effect_size": d
            }

        out_path = os.path.join(self.output_dir, "metrics", "statistical_validation.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

