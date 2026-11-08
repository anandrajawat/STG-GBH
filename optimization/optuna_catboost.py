"""
Optuna TPE Hyperparameter Optimization for CatBoost in STG-GBH.
Objective: Macro-F1 with early stopping on chronological validation split.
"""

import os
import json
import optuna
import numpy as np
from typing import Dict, Any
from sklearn.metrics import f1_score, average_precision_score, log_loss
try:
    from models.catboost_model import CatBoostModel
except ImportError:
    from ..models.catboost_model import CatBoostModel


def run_optuna_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    output_dir: str = "results/optimization",
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs Optuna TPE search for CatBoost and exports best configuration."""
    os.makedirs(output_dir, exist_ok=True)
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 150, 600, step=50),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "verbose": False
        }

        model = CatBoostModel(params=params, task="binary", random_state=random_state)
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
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_result = {
        "best_trial_number": study.best_trial.number,
        "best_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "all_trials": trial_records
    }

    out_path = os.path.join(output_dir, "catboost_best.json")
    with open(out_path, "w") as f:
        json.dump(best_result, f, indent=2)

    return best_result
