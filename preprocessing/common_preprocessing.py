"""
Common preprocessing adapter and transformers for STG-GBH.
Strictly adheres to training-only fitting to guarantee zero data leakage.
"""

import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import RobustScaler


class LeakageFreePreprocessor:
    """
    Transforms network flow features using train-only fitted statistics.
    Ensures no distribution or scale information from eval sets leaks into training.
    """
    def __init__(self, feature_cols: List[str], scale_method: str = "robust"):
        self.feature_cols = list(feature_cols)
        self.scale_method = scale_method
        self.fill_medians: Dict[str, float] = {}
        self.scaler = RobustScaler(quantile_range=(5.0, 95.0)) if scale_method == "robust" else None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "LeakageFreePreprocessor":
        """Fit imputation medians and scaler solely on the training dataframe."""
        num_df = train_df[self.feature_cols].copy()

        self.fill_medians = {}
        for col in self.feature_cols:
            if col in num_df.columns and pd.api.types.is_numeric_dtype(num_df[col]):
                series = num_df[col].replace([np.inf, -np.inf], np.nan)
                med = float(series.median())
                if np.isnan(med) or np.isinf(med):
                    med = 0.0
                self.fill_medians[col] = med
                num_df[col] = series.fillna(med).clip(-1e12, 1e12)
            else:
                num_df[col] = 0.0
                self.fill_medians[col] = 0.0

        if self.scaler is not None:
            self.scaler.fit(num_df[self.feature_cols].values)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame using strictly the fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform()!")

        num_df = df[self.feature_cols].copy()
        for col in self.feature_cols:
            if col in num_df.columns and pd.api.types.is_numeric_dtype(num_df[col]):
                num_df[col] = (
                    num_df[col]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.fill_medians.get(col, 0.0))
                    .clip(-1e12, 1e12)
                )
            else:
                num_df[col] = self.fill_medians.get(col, 0.0)

        X = num_df[self.feature_cols].values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X = np.clip(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        return X

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LeakageFreePreprocessor":
        return joblib.load(filepath)


def chronological_split(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions data chronologically to evaluate temporal generalization.
    If time_col is not provided, maintains the natural capture sequence order.
    """
    if time_col and time_col in df.columns:
        sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
    else:
        sorted_df = df.reset_index(drop=True)

    n = len(sorted_df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[n_train:n_val].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_val:].copy().reset_index(drop=True)

    return train_df, val_df, test_df
