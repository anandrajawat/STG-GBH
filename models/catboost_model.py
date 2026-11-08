"""
CatBoost model wrapper for STG-GBH.
Optimized for tabular cyber intrusion features with gradient-based regularization.
"""

import joblib
import numpy as np
from catboost import CatBoostClassifier
from typing import Dict, Any, Optional, Tuple


class CatBoostModel:
    """
    CatBoost classifier with symmetric tree structure and native class weighting.
    """
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        task: str = "binary",
        n_classes: int = 2,
        random_state: int = 42
    ):
        self.task = task
        self.n_classes = n_classes
        self.random_state = random_state

        default_params = {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "border_count": 128,
            "random_seed": random_state,
            "thread_count": -1,
            "verbose": False
        }
        if task == "binary":
            default_params["loss_function"] = "Logloss"
            default_params["eval_metric"] = "Logloss"
        else:
            default_params["loss_function"] = "MultiClass"
            default_params["eval_metric"] = "MultiClass"

        if params:
            default_params.update(params)
        self.params = default_params
        self.model = CatBoostClassifier(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        # Auto-balance binary weights if requested
        auto_weights = None
        if self.task == "binary" and sample_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                auto_weights = [1.0, float(np.clip(neg_count / float(pos_count), 0.2, 20.0))]
                self.model.set_params(class_weights=auto_weights)

        eval_data = eval_set if eval_set is not None else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_data,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
            sample_weight=sample_weight,
            verbose=verbose
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).ravel()

    def get_feature_importances(self) -> np.ndarray:
        return self.model.get_feature_importance()

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)

