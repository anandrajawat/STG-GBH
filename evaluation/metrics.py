"""
Comprehensive scientific evaluation metrics suite for STG-GBH.
Calculates Accuracy, Precision, Recall, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC,
FPR, FNR, per-class breakdowns, and confusion matrices.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = "binary",
    target_names: Optional[list] = None
) -> Dict[str, Any]:
    """
    Computes complete academic evaluation metrics dictionary.
    """
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    if task == "binary":
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion matrix breakdown
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            fpr, fnr = 0.0, 0.0

        roc_auc = 0.5
        pr_auc = 0.0
        if y_proba is not None:
            # Check if 1D or 2D
            prob_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            if len(np.unique(y_true)) > 1:
                try:
                    roc_auc = float(roc_auc_score(y_true, prob_pos))
                    pr_auc = float(average_precision_score(y_true, prob_pos))
                except Exception:
                    pass

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": fpr,
            "fnr": fnr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "fp_per_10k_benign": fpr * 10000.0,
            "fn_per_10k_malicious": fnr * 10000.0,
            "confusion_matrix": cm.tolist()
        }
    else:
        # Multiclass evaluation
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # One-vs-Rest FPR and FNR
        num_classes = cm.shape[0]
        ovr_fpr = []
        ovr_fnr = []
        for i in range(num_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - (tp_i + fn_i + fp_i)
            ovr_fpr.append(fp_i / float(fp_i + tn_i) if (fp_i + tn_i) > 0 else 0.0)
            ovr_fnr.append(fn_i / float(fn_i + tp_i) if (fn_i + tp_i) > 0 else 0.0)

        roc_auc = 0.5
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            except Exception:
                pass

        return {
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "macro_fpr": float(np.mean(ovr_fpr)),
            "macro_fnr": float(np.mean(ovr_fnr)),
            "ovr_fpr_per_class": [float(x) for x in ovr_fpr],
            "ovr_fnr_per_class": [float(x) for x in ovr_fnr],
            "confusion_matrix": cm.tolist()
        }

