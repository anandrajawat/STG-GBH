"""
LightGBM model wrapper for STG-GBH.
Histogram-based gradient boosted trees with leaf-wise splitting and early stopping.
"""

import joblib
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional, Tuple


class LightGBMModel:
    """
    LightGBM classifier with leaf-wise growth and fast histogram binning.
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
            "n_estimators": 500,
            "num_leaves": 31,
            "max_depth": 7,
            "learning_rate": 0.05,
            "max_bin": 128,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1
        }
        if task == "binary":
            default_params["objective"] = "binary"
        else:
            default_params["objective"] = "multiclass"
            default_params["num_class"] = n_classes

        if params:
            default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)

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
        if self.task == "binary" and sample_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                scale = float(np.clip(neg_count / float(pos_count), 0.2, 20.0))
                self.model.set_params(scale_pos_weight=scale)

        callbacks = []
        if eval_set is not None and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))

        self.model.fit(
            X_train,
            y_train,
            eval_set=[eval_set] if eval_set is not None else None,
            sample_weight=sample_weight,
            callbacks=callbacks
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)

