"""
Optuna TPE Hyperparameter Optimization Runner for STG-GBH.
Executes systematic search across XGBoost, CatBoost, and LightGBM.
Saves trial logs and best parameters to results/optimization/.
"""

import os
import json
from preprocessing.dataset_adapters.cse_cic_ids2018 import CSECICIDS2018Adapter
from preprocessing.common_preprocessing import LeakageFreePreprocessor
from preprocessing.duplicate_detection import purge_cross_split_leakage
from optimization.optuna_xgb import run_optuna_xgboost
from optimization.optuna_catboost import run_optuna_catboost
from optimization.optuna_lightgbm import run_optuna_lightgbm


def run_all_optimizations(data_dir: str = None, n_trials: int = 15, output_dir: str = "results/optimization"):
    print("=================================================================")
    print("  STARTING OPTUNA TPE HYPERPARAMETER OPTIMIZATION               ")
    print("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/CSE_CIC_IDS2018")

    # Load chronological splits
    adapter = CSECICIDS2018Adapter(data_dir)
    train_df, val_df, _, raw_feats = adapter.load_chronological_splits(
        samples_per_day=3000, train_days=6, val_days=2, test_days=2
    )

    feature_cols = [c for c in raw_feats if c not in ["Label", "session_id", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]]
    val_clean = purge_cross_split_leakage(train_df, val_df, feature_cols[:20])

    prep = LeakageFreePreprocessor(feature_cols, scale_method="robust")
    X_train = prep.fit_transform(train_df)
    y_train = train_df["is_attack"].values.astype(int)

    X_val = prep.transform(val_clean)
    y_val = val_clean["is_attack"].values.astype(int)

    # 1. XGBoost Optuna
    print("\n[1/3] Running Optuna TPE for XGBoost...")
    xgb_best = run_optuna_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials, output_dir=output_dir)
    print(f"XGBoost Best Macro-F1: {xgb_best['best_macro_f1']:.4f}")

    # 2. CatBoost Optuna
    print("\n[2/3] Running Optuna TPE for CatBoost...")
    cat_best = run_optuna_catboost(X_train, y_train, X_val, y_val, n_trials=n_trials, output_dir=output_dir)
    print(f"CatBoost Best Macro-F1: {cat_best['best_macro_f1']:.4f}")

    # 3. LightGBM Optuna
    print("\n[3/3] Running Optuna TPE for LightGBM...")
    lgb_best = run_optuna_lightgbm(X_train, y_train, X_val, y_val, n_trials=n_trials, output_dir=output_dir)
    print(f"LightGBM Best Macro-F1: {lgb_best['best_macro_f1']:.4f}")

    # Build Optimization Summary Table
    summary = [
        {"Model": "XGBoost", "Best_Macro_F1": xgb_best["best_macro_f1"], "Best_Trial": xgb_best["best_trial_number"], "Best_Params": str(xgb_best["best_params"])},
        {"Model": "CatBoost", "Best_Macro_F1": cat_best["best_macro_f1"], "Best_Trial": cat_best["best_trial_number"], "Best_Params": str(cat_best["best_params"])},
        {"Model": "LightGBM", "Best_Macro_F1": lgb_best["best_macro_f1"], "Best_Trial": lgb_best["best_trial_number"], "Best_Params": str(lgb_best["best_params"])}
    ]
    import pandas as pd
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "hyperparameter_optimization_summary.csv"), index=False)
    print("\nHyperparameter optimization completed and exported successfully!")


if __name__ == "__main__":
    run_all_optimizations()
