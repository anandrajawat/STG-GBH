"""
Optuna TPE Hyperparameter Optimization for LightGBM in STG-GBH.
Objective: Macro-F1 with Median Pruning on chronological validation split.
"""

import os
import json
import optuna
import numpy as np
from typing import Dict, Any
from sklearn.metrics import f1_score, average_precision_score, log_loss
try:
    from models.lightgbm_model import LightGBMModel
except ImportError:
    from ..models.lightgbm_model import LightGBMModel


def run_optuna_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    output_dir: str = "results/optimization",
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs Optuna TPE search for LightGBM and exports best configuration."""
    os.makedirs(output_dir, exist_ok=True)
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_bin": trial.suggest_int("max_bin", 63, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 15.0),
            "verbose": -1
        }

        model = LightGBMModel(params=params, task="binary", random_state=random_state)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        preds = model.predict(X_val)
        probas = model.predict_proba(X_val)

        macro_f1 = float(f1_score(y_val, preds, average="macro"))
        pr_auc = float(average_precision_score(y_val, probas[:, 1]))
        loss = float(log_loss(y_val, probas))

        trial_records.append({
            "trial_number": trial.number,
            "params": params,
            "macro_f1": macro_f1,
            "pr_auc": pr_auc,
            "log_loss": loss
        })

        return macro_f1

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_result = {
        "best_trial_number": study.best_trial.number,
        "best_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "all_trials": trial_records
    }

    out_path = os.path.join(output_dir, "lightgbm_best.json")
    with open(out_path, "w") as f:
        json.dump(best_result, f, indent=2)

    return best_result
