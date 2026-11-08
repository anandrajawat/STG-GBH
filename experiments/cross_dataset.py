"""
Cross-Dataset Generalization Evaluation for STG-GBH: CSE-CIC-IDS2018 -> CIC-IDS2017.
Evaluates:
  A. Zero-shot transfer (models trained on 2018 evaluated directly on 2017)
  B. Domain-adapted transfer (calibrated ensemble & threshold on target validation split)
  C. In-domain reference baseline trained and tested on CIC-IDS2017
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from preprocessing.dataset_adapters.cse_cic_ids2018 import CSECICIDS2018Adapter
from preprocessing.dataset_adapters.cic_ids2017 import CICIDS2017Adapter
from preprocessing.common_preprocessing import LeakageFreePreprocessor
from preprocessing.duplicate_detection import purge_cross_split_leakage
from preprocessing.feature_engineering import (
    compute_network_features,
    compute_causal_temporal_features,
    compute_topological_graph_features
)
from models.stacking_ensemble import OOFStackingEnsemble
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from optimization.threshold_optimization import optimize_threshold_constrained
from evaluation.metrics import compute_comprehensive_metrics


class CrossDatasetExperiment:
    """
    Evaluates transferability across network environments: CSE-CIC-IDS2018 to CIC-IDS2017.
    """
    def __init__(
        self,
        c18_dir: str = "datasets/CSE_CIC_IDS2018",
        c17_dir: str = "datasets/CIC_IDS2017",
        output_dir: str = "results",
        samples_per_file: int = 4000,
        random_state: int = 42
    ):
        self.c18_dir = c18_dir
        self.c17_dir = c17_dir
        self.output_dir = output_dir
        self.samples_per_file = samples_per_file
        self.random_state = random_state
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        print("\n=================================================================")
        print("  STARTING CROSS-DATASET EXPERIMENT: CSE-2018 -> CIC-2017         ")
        print("=================================================================")

        # 1. Load CSE-CIC-IDS2018 Source Training Data
        print("\n[Step 1/5] Loading Source Dataset (CSE-CIC-IDS2018)...")
        c18_adapter = CSECICIDS2018Adapter(self.c18_dir)
        train_df, val_df, _, raw_feats = c18_adapter.load_chronological_splits(
            samples_per_day=self.samples_per_file, train_days=6, val_days=2, test_days=2
        )

        train_comb = pd.concat([
            train_df,
            compute_network_features(train_df),
            compute_causal_temporal_features(train_df),
            compute_topological_graph_features(train_df)
        ], axis=1)

        val_comb = pd.concat([
            val_df,
            compute_network_features(val_df),
            compute_causal_temporal_features(val_df),
            compute_topological_graph_features(val_df)
        ], axis=1)

        # 2. Load CIC-IDS2017 Target Dataset
        print("\n[Step 2/5] Loading Target Dataset (CIC-IDS2017)...")
        c17_adapter = CICIDS2017Adapter(self.c17_dir)
        c17_full, _ = c17_adapter.load_all(samples_per_file=self.samples_per_file)
        c17_full["session_id"] = c17_full["session_file"]

        c17_comb = pd.concat([
            c17_full,
            compute_network_features(c17_full),
            compute_causal_temporal_features(c17_full),
            compute_topological_graph_features(c17_full)
        ], axis=1)

        common_features = [
            c for c in train_comb.columns
            if c in c17_comb.columns and c not in [
                "Label", "session_id", "session_file", "source_dataset",
                "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"
            ]
        ]
        print(f"Common Feature Intersection: {len(common_features)} aligned features.")

        # Chronological split within each CIC-IDS2017 file so all attack families exist in target-train/val and target-test
        val_parts, test_parts = [], []
        for s_file, grp_df in c17_comb.groupby("session_file", sort=False):
            n_g = len(grp_df)
            cut = int(0.50 * n_g)
            val_parts.append(grp_df.iloc[:cut])
            test_parts.append(grp_df.iloc[cut:])
        c17_val = pd.concat(val_parts, ignore_index=True)
        c17_test = pd.concat(test_parts, ignore_index=True)
        c17_test = purge_cross_split_leakage(train_comb, c17_test, raw_feats[:20])

        # 3. Fit Preprocessor ONLY on Source 2018 Training Data
        print("\n[Step 3/5] Fitting Source Preprocessor...")
        preprocessor = LeakageFreePreprocessor(common_features, scale_method="robust")
        X_src_train = preprocessor.fit_transform(train_comb)
        y_src_train = train_comb["is_attack"].values.astype(int)

        X_src_val = preprocessor.transform(val_comb)
        y_src_val = val_comb["is_attack"].values.astype(int)

        X_tgt_test = preprocessor.transform(c17_test)
        y_tgt_test = c17_test["is_attack"].values.astype(int)

        X_tgt_val = preprocessor.transform(c17_val)
        y_tgt_val = c17_val["is_attack"].values.astype(int)

        # 4. Train Models on Source (2018)
        print("\n[Step 4/5] Training Source Models...")
        xgb_src = XGBoostModel(task="binary", random_state=self.random_state).fit(
            X_src_train, y_src_train, eval_set=(X_src_val, y_src_val), verbose=False
        )
        cat_src = CatBoostModel(task="binary", random_state=self.random_state).fit(
            X_src_train, y_src_train, eval_set=(X_src_val, y_src_val), verbose=False
        )
        lgb_src = LightGBMModel(task="binary", random_state=self.random_state).fit(
            X_src_train, y_src_train, eval_set=(X_src_val, y_src_val), verbose=False
        )
        stack_src = OOFStackingEnsemble(task="binary", n_splits=5, random_state=self.random_state).fit(
            X_src_train, y_src_train, eval_set=(X_src_val, y_src_val), verbose=False
        )

        # 5. Evaluate: A. Zero-Shot Transfer
        print("\n[Step 5/5] Evaluating Transfer Performance...")
        p_xgb_tgt = xgb_src.predict_proba(X_tgt_test)
        p_cat_tgt = cat_src.predict_proba(X_tgt_test)
        p_lgb_tgt = lgb_src.predict_proba(X_tgt_test)
        p_stack_tgt = stack_src.predict_proba(X_tgt_test)
        p_voting_tgt = (p_xgb_tgt + p_cat_tgt + p_lgb_tgt) / 3.0

        self.results["ZeroShot_XGBoost"] = compute_comprehensive_metrics(
            y_tgt_test, (p_xgb_tgt[:, 1] >= 0.5).astype(int), p_xgb_tgt
        )
        self.results["ZeroShot_CatBoost"] = compute_comprehensive_metrics(
            y_tgt_test, (p_cat_tgt[:, 1] >= 0.5).astype(int), p_cat_tgt
        )
        self.results["ZeroShot_LightGBM"] = compute_comprehensive_metrics(
            y_tgt_test, (p_lgb_tgt[:, 1] >= 0.5).astype(int), p_lgb_tgt
        )
        self.results["ZeroShot_SoftVoting"] = compute_comprehensive_metrics(
            y_tgt_test, (p_voting_tgt[:, 1] >= 0.5).astype(int), p_voting_tgt
        )
        self.results["ZeroShot_OOF_Stacking"] = compute_comprehensive_metrics(
            y_tgt_test, (p_stack_tgt[:, 1] >= 0.5).astype(int), p_stack_tgt
        )

        # B. Domain-Adapted Transfer: Calibrate meta-learner on target validation split
        from sklearn.linear_model import LogisticRegression
        meta_val_feats = stack_src._build_meta_features(X_tgt_val)
        meta_test_feats = stack_src._build_meta_features(X_tgt_test)
        adapted_meta = LogisticRegression(C=2.0, max_iter=1000, random_state=self.random_state)
        adapted_meta.fit(meta_val_feats, y_tgt_val)
        p_adapted_val = adapted_meta.predict_proba(meta_val_feats)
        p_adapted_test = adapted_meta.predict_proba(meta_test_feats)
        opt_ad = optimize_threshold_constrained(y_tgt_val, p_adapted_val[:, 1], target_max_fpr=0.02, target_max_fnr=0.02)
        self.results["Domain_Adapted_Stacking"] = compute_comprehensive_metrics(
            y_tgt_test, (p_adapted_test[:, 1] >= opt_ad["chosen_threshold"]).astype(int), p_adapted_test
        )

        # C. In-Domain Reference on CIC-IDS2017 (Upper Bound)
        print(" -> In-Domain Reference Benchmark on CIC-IDS2017...")
        tgt_prep = LeakageFreePreprocessor(common_features, scale_method="robust")
        X_tgt_tr = tgt_prep.fit_transform(c17_val)
        X_tgt_te_in = tgt_prep.transform(c17_test)

        stack_indomain = OOFStackingEnsemble(task="binary", n_splits=5, random_state=self.random_state)
        stack_indomain.fit(X_tgt_tr, y_tgt_val, verbose=False)
        p_indomain = stack_indomain.predict_proba(X_tgt_te_in)
        self.results["InDomain_Reference_CIC2017"] = compute_comprehensive_metrics(
            y_tgt_test, (p_indomain[:, 1] >= 0.5).astype(int), p_indomain
        )

        out_file = os.path.join(self.output_dir, "metrics", "cross_dataset_metrics.json")
        with open(out_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved cross-dataset metrics to {out_file}")
        return self.results


if __name__ == "__main__":
    exp = CrossDatasetExperiment()
    results = exp.run()
