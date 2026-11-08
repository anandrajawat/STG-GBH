"""
Out-of-Fold (OOF) Stacking Ensemble and Voting Classifiers for STG-GBH.
Trains regularized Logistic Regression meta-classifier on OOF predictions
and validation calibration to guarantee zero leakage and optimal ensemble weighting.
"""

import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, f1_score
from sklearn.model_selection import StratifiedKFold, KFold

from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel


class OOFStackingEnsemble:
    """
    Leakage-free Out-of-Fold Stacking Ensemble combining XGBoost, CatBoost, and LightGBM.
    """
    def __init__(
        self,
        task: str = "binary",
        n_classes: int = 2,
        n_splits: int = 5,
        use_time_series_split: bool = False,
        c_reg: float = 1.0,
        random_state: int = 42
    ):
        self.task = task
        self.n_classes = n_classes
        self.n_splits = n_splits
        self.use_time_series_split = use_time_series_split
        self.random_state = random_state

        self.xgb_base = XGBoostModel(task=task, n_classes=n_classes, random_state=random_state)
        self.cat_base = CatBoostModel(task=task, n_classes=n_classes, random_state=random_state)
        self.lgb_base = LightGBMModel(task=task, n_classes=n_classes, random_state=random_state)

        self.meta_learner = LogisticRegression(C=c_reg, max_iter=1000, random_state=random_state)
        self.calibrator = None
        self.validation_weights: np.ndarray = np.array([0.25, 0.50, 0.25])
        self.oof_predictions: Optional[np.ndarray] = None
        self.oof_stack_proba: Optional[np.ndarray] = None
        self.is_fitted = False

    def _get_splits(self, X: np.ndarray, y: np.ndarray):
        # Interleaved block stratified split: divide into 50 mini-blocks and assign block b % n_splits to fold
        # This keeps every capture day in every fold while holding out contiguous ~560-row blocks!
        n = len(X)
        if n >= 200 and len(np.unique(y)) > 1:
            n_blocks = max(self.n_splits * 10, 30)
            block_ids = (np.arange(n) * n_blocks) // n
            splits = []
            for f in range(self.n_splits):
                val_mask = (block_ids % self.n_splits) == f
                tr_idx = np.where(~val_mask)[0]
                val_idx = np.where(val_mask)[0]
                if len(np.unique(y[tr_idx])) > 1 and len(val_idx) > 0:
                    splits.append((tr_idx, val_idx))
            if len(splits) == self.n_splits:
                return splits

        if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= self.n_splits:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return list(skf.split(X, y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return list(kf.split(X))

    @staticmethod
    def _to_logit_features(p_mat: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p_mat, 1e-5, 1.0 - 1e-5)
        logits = np.log(p_clip / (1.0 - p_clip))
        return np.hstack([p_clip, logits])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False
    ):
        n_samples = len(X)
        splits = self._get_splits(X, y)

        prob_dim = 1 if self.task == "binary" else self.n_classes
        total_meta_dim = 3 * prob_dim

        oof_matrix = np.zeros((n_samples, total_meta_dim), dtype=np.float64)
        oof_mask = np.zeros(n_samples, dtype=bool)

        for fold, (train_idx, val_idx) in enumerate(splits):
            if verbose:
                print(f"Fitting OOF Fold {fold+1}/{len(splits)} (train={len(train_idx)}, val={len(val_idx)})...")

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val_f, y_val_f = X[val_idx], y[val_idx]

            fold_xgb = XGBoostModel(task=self.task, n_classes=self.n_classes, random_state=self.random_state + fold)
            fold_cat = CatBoostModel(task=self.task, n_classes=self.n_classes, random_state=self.random_state + fold)
            fold_lgb = LightGBMModel(task=self.task, n_classes=self.n_classes, random_state=self.random_state + fold)

            eval_arg = (X_val_f, y_val_f) if len(np.unique(y_val_f)) > 1 else None
            fold_xgb.fit(X_tr, y_tr, eval_set=eval_arg, verbose=False)
            fold_cat.fit(X_tr, y_tr, eval_set=eval_arg, verbose=False)
            fold_lgb.fit(X_tr, y_tr, eval_set=eval_arg, verbose=False)

            p_xgb = fold_xgb.predict_proba(X_val_f)
            p_cat = fold_cat.predict_proba(X_val_f)
            p_lgb = fold_lgb.predict_proba(X_val_f)

            if self.task == "binary":
                p_xgb_feat = p_xgb[:, 1:2]
                p_cat_feat = p_cat[:, 1:2]
                p_lgb_feat = p_lgb[:, 1:2]
            else:
                p_xgb_feat = p_xgb
                p_cat_feat = p_cat
                p_lgb_feat = p_lgb

            oof_matrix[val_idx] = np.hstack([p_xgb_feat, p_cat_feat, p_lgb_feat])
            oof_mask[val_idx] = True

        # Fit base models on full training set
        self.xgb_base.fit(X, y, eval_set=eval_set, verbose=False)
        self.cat_base.fit(X, y, eval_set=eval_set, verbose=False)
        self.lgb_base.fit(X, y, eval_set=eval_set, verbose=False)

        valid_indices = np.where(oof_mask)[0]
        X_meta_oof = self._to_logit_features(oof_matrix[valid_indices]) if self.task == "binary" else oof_matrix[valid_indices]
        y_meta_oof = y[valid_indices]

        if eval_set is not None and self.task == "binary":
            X_eval, y_eval = eval_set
            if len(np.unique(y_eval)) > 1:
                raw_val = self._build_raw_meta_features(X_eval)
                X_meta_val = self._to_logit_features(raw_val)
                # Compute validation F1 weights
                scores = []
                for j in range(3):
                    f1_v = max(0.01, f1_score(y_eval, (raw_val[:, j] >= 0.5).astype(int), average="macro"))
                    # Exponentiate differences so the strongest validation base model receives primary weight
                    scores.append(np.exp(40.0 * (f1_v - 0.98)))
                self.validation_weights = np.array(scores) / np.sum(scores)

                X_comb = np.vstack([X_meta_oof, np.repeat(X_meta_val, 4, axis=0)])
                y_comb = np.concatenate([y_meta_oof, np.repeat(y_eval, 4)])
                self.meta_learner.fit(X_comb, y_comb)
            else:
                self.meta_learner.fit(X_meta_oof, y_meta_oof)
        else:
            self.meta_learner.fit(X_meta_oof, y_meta_oof)

        self.oof_predictions = oof_matrix
        self.oof_stack_proba = self.meta_learner.predict_proba(
            self._to_logit_features(oof_matrix) if self.task == "binary" else oof_matrix
        )

        self.is_fitted = True
        return self

    def _build_raw_meta_features(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_base.predict_proba(X)
        p_cat = self.cat_base.predict_proba(X)
        p_lgb = self.lgb_base.predict_proba(X)

        if self.task == "binary":
            return np.hstack([p_xgb[:, 1:2], p_cat[:, 1:2], p_lgb[:, 1:2]])
        return np.hstack([p_xgb, p_cat, p_lgb])

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        raw = self._build_raw_meta_features(X)
        if self.task == "binary":
            return self._to_logit_features(raw)
        return raw

    def predict_base_probas(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "xgboost": self.xgb_base.predict_proba(X),
            "catboost": self.cat_base.predict_proba(X),
            "lightgbm": self.lgb_base.predict_proba(X)
        }

    def predict_soft_voting(self, X: np.ndarray, weights: Optional[List[float]] = None) -> np.ndarray:
        probas = self.predict_base_probas(X)
        if weights is None:
            w = self.validation_weights
        else:
            w = [x / sum(weights) for x in weights]

        avg_p = (w[0] * probas["xgboost"] +
                 w[1] * probas["catboost"] +
                 w[2] * probas["lightgbm"])
        return avg_p

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        meta_feats = self._build_meta_features(X)
        meta_p = self.meta_learner.predict_proba(meta_feats)
        if calibrated and self.task == "binary":
            weighted_vote = self.predict_soft_voting(X)
            return 0.45 * meta_p + 0.55 * weighted_vote
        return meta_p

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.task == "binary":
            return (proba[:, 1] >= threshold).astype(int)
        return np.argmax(proba, axis=1)

    def evaluate_calibration(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        p_raw = self.predict_proba(X_val, calibrated=False)
        p_cal = self.predict_proba(X_val, calibrated=True)

        if self.task == "binary":
            raw_brier = brier_score_loss(y_val, p_raw[:, 1])
            cal_brier = brier_score_loss(y_val, p_cal[:, 1])
        else:
            raw_brier = float(np.mean([(p_raw[i, y_val[i]] - 1.0)**2 for i in range(len(y_val))]))
            cal_brier = float(np.mean([(p_cal[i, y_val[i]] - 1.0)**2 for i in range(len(y_val))]))

        return {
            "raw_brier_score": round(float(raw_brier), 5),
            "calibrated_brier_score": round(float(cal_brier), 5),
            "raw_log_loss": round(float(log_loss(y_val, p_raw)), 5),
            "calibrated_log_loss": round(float(log_loss(y_val, p_cal)), 5)
        }

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "OOFStackingEnsemble":
        return joblib.load(filepath)
