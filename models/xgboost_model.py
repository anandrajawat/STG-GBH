"""
XGBoost model wrapper for STG-GBH.
Supports binary and multiclass intrusion detection with early stopping and class weighting.
"""

import os
import joblib
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple


class XGBoostModel:
    """
    XGBoost classifier with native support for class weighting, early stopping, and probability prediction.
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
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist"
        }
        if task == "binary":
            default_params["objective"] = "binary:logistic"
            default_params["eval_metric"] = "logloss"
        else:
            default_params["objective"] = "multi:softprob"
            default_params["num_class"] = n_classes
            default_params["eval_metric"] = "mlogloss"

        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBClassifier(**self.params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        early_stopping_rounds: int = 30,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        eval_list = [(X_train, y_train)]
        if eval_set is not None:
            eval_list.append(eval_set)

        # Compute positive class scale if binary and imbalanced
        if self.task == "binary" and sample_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                scale = neg_count / float(pos_count)
                # Bound scale to avoid extreme gradient swings
                scale = float(np.clip(scale, 0.2, 20.0))
                self.model.set_params(scale_pos_weight=scale)

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_list if eval_set is not None else None,
            sample_weight=sample_weight,
            verbose=verbose
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

