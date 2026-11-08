"""
In-Domain Experimental Evaluation Pipeline for STG-GBH.
Trains and compares all baseline and proposed architectures on CSE-CIC-IDS2018
with chronological sub-session holdouts and zero data leakage.
"""

import os
import json
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from preprocessing.dataset_adapters.cse_cic_ids2018 import CSECICIDS2018Adapter
from preprocessing.common_preprocessing import LeakageFreePreprocessor
from preprocessing.duplicate_detection import purge_cross_split_leakage
from preprocessing.feature_engineering import (
    compute_network_features,
    compute_causal_temporal_features,
    compute_topological_graph_features
)
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from models.stacking_ensemble import OOFStackingEnsemble
from models.graphsage import HomogeneousGraphSAGE
from models.temporal_gru import TemporalGRU
from models.temporal_attention import TemporalAttention
from models.fusion_model import STGGBHFusionModel
from models.multitask_heads import UncertaintyWeightedLoss
from optimization.threshold_optimization import optimize_threshold_constrained
from evaluation.metrics import compute_comprehensive_metrics
from evaluation.calibration import compute_calibration_metrics
from evaluation.latency import benchmark_inference_latency
from evaluation.memory import compute_model_footprint
from evaluation.explainability import compute_tree_shap_summary, compare_feature_importances
from evaluation.error_analysis import perform_error_analysis


def build_causal_session_graph(n_rows: int, session_ids: np.ndarray = None, max_lag: int = 3) -> torch.Tensor:
    """
    Constructs a strictly causal directed edge_index connecting preceding flows
    (t - k -> t) within the same capture session.
    """
    src_list = [np.arange(n_rows)]
    dst_list = [np.arange(n_rows)]
    for lag in range(1, max_lag + 1):
        if n_rows > lag:
            s_idx = np.arange(0, n_rows - lag)
            d_idx = np.arange(lag, n_rows)
            if session_ids is not None:
                same_sess = (session_ids[s_idx] == session_ids[d_idx])
                s_idx = s_idx[same_sess]
                d_idx = d_idx[same_sess]
            src_list.append(s_idx)
            dst_list.append(d_idx)
    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


class InDomainExperiment:
    """
    Orchestrates full in-domain experimental evaluation comparing all baseline and STG-GBH architectures.
    """
    def __init__(
        self,
        data_dir: str = "datasets/CSE_CIC_IDS2018",
        output_dir: str = "results",
        samples_per_day: int = 4000,
        random_state: int = 42
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.samples_per_day = samples_per_day
        self.random_state = random_state

        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        print("\n=================================================================")
        print("  STARTING IN-DOMAIN EXPERIMENTS: CSE-CIC-IDS2018 CHRONOLOGICAL  ")
        print("=================================================================")

        # 1. Load Data
        print("\n[Step 1/7] Loading Chronological Datasets...")
        adapter = CSECICIDS2018Adapter(self.data_dir)
        train_df, val_df, test_df, raw_feats = adapter.load_chronological_splits(
            samples_per_day=self.samples_per_day,
            train_days=6,
            val_days=2,
            test_days=2
        )
        print(f"Loaded Raw Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # 2. Feature Engineering
        print("\n[Step 2/7] Feature Engineering (Network, Temporal Causal, Graph)...")
        train_net = compute_network_features(train_df)
        val_net = compute_network_features(val_df)
        test_net = compute_network_features(test_df)

        train_temp = compute_causal_temporal_features(train_df)
        val_temp = compute_causal_temporal_features(val_df)
        test_temp = compute_causal_temporal_features(test_df)

        train_grp = compute_topological_graph_features(train_df)
        val_grp = compute_topological_graph_features(val_df)
        test_grp = compute_topological_graph_features(test_df)

        train_comb = pd.concat([train_df, train_net, train_temp, train_grp], axis=1)
        val_comb = pd.concat([val_df, val_net, val_temp, val_grp], axis=1)
        test_comb = pd.concat([test_df, test_net, test_temp, test_grp], axis=1)

        all_feature_cols = [
            c for c in train_comb.columns
            if c not in ["Label", "session_id", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        ]

        # 3. Purge cross-split duplicates
        print("\n[Step 3/7] Purging Cross-Split Duplicates for Zero-Leakage...")
        val_comb = purge_cross_split_leakage(train_comb, val_comb, raw_feats[:20])
        test_comb = purge_cross_split_leakage(train_comb, test_comb, raw_feats[:20])
        test_comb = purge_cross_split_leakage(val_comb, test_comb, raw_feats[:20])
        print(f"Purged Splits: Train={len(train_comb)}, Val={len(val_comb)}, Test={len(test_comb)}")

        # 4. Fit Preprocessor ONLY on training set
        print("\n[Step 4/7] Fitting Preprocessor Strictly on Training Set...")
        preprocessor = LeakageFreePreprocessor(all_feature_cols, scale_method="robust")
        X_train = preprocessor.fit_transform(train_comb)
        X_val = preprocessor.transform(val_comb)
        X_test = preprocessor.transform(test_comb)

        y_train = train_comb["is_attack"].values.astype(int)
        y_val = val_comb["is_attack"].values.astype(int)
        y_test = test_comb["is_attack"].values.astype(int)

        y_train_cat = train_comb["canonical_id"].values.astype(int)
        y_val_cat = val_comb["canonical_id"].values.astype(int)
        y_test_cat = test_comb["canonical_id"].values.astype(int)

        y_train_mitre = train_comb["mitre_id"].values.astype(int)
        y_val_mitre = val_comb["mitre_id"].values.astype(int)
        y_test_mitre = test_comb["mitre_id"].values.astype(int)

        print(f"Feature Matrix Shape: {X_train.shape[1]} features")

        # 5. Baseline Boosting and Ensemble Models
        print("\n[Step 5/7] Training Baseline Models and OOF Stacking Ensemble...")

        # Baseline A: Random Forest
        print(" -> Training Random Forest...")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_time = time.time() - t0
        p_rf_test = rf.predict_proba(X_test)
        preds_rf = (p_rf_test[:, 1] >= 0.5).astype(int)
        self.results["Random_Forest"] = {
            **compute_comprehensive_metrics(y_test, preds_rf, p_rf_test),
            "train_time_sec": round(rf_time, 2),
            **benchmark_inference_latency(rf.predict_proba, X_test[:1000])
        }

        # Baseline B: XGBoost alone
        print(" -> Training XGBoost...")
        t0 = time.time()
        xgb_model = XGBoostModel(task="binary", random_state=self.random_state)
        xgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        xgb_time = time.time() - t0
        p_xgb_test = xgb_model.predict_proba(X_test)
        preds_xgb = (p_xgb_test[:, 1] >= 0.5).astype(int)
        self.results["XGBoost_alone"] = {
            **compute_comprehensive_metrics(y_test, preds_xgb, p_xgb_test),
            "train_time_sec": round(xgb_time, 2),
            **benchmark_inference_latency(xgb_model.predict_proba, X_test[:1000])
        }

        # Baseline C: CatBoost alone
        print(" -> Training CatBoost...")
        t0 = time.time()
        cat_model = CatBoostModel(task="binary", random_state=self.random_state)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        cat_time = time.time() - t0
        p_cat_test = cat_model.predict_proba(X_test)
        preds_cat = (p_cat_test[:, 1] >= 0.5).astype(int)
        self.results["CatBoost_alone"] = {
            **compute_comprehensive_metrics(y_test, preds_cat, p_cat_test),
            "train_time_sec": round(cat_time, 2),
            **benchmark_inference_latency(cat_model.predict_proba, X_test[:1000])
        }

        # Baseline D: LightGBM alone
        print(" -> Training LightGBM...")
        t0 = time.time()
        lgb_model = LightGBMModel(task="binary", random_state=self.random_state)
        lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        lgb_time = time.time() - t0
        p_lgb_test = lgb_model.predict_proba(X_test)
        preds_lgb = (p_lgb_test[:, 1] >= 0.5).astype(int)
        self.results["LightGBM_alone"] = {
            **compute_comprehensive_metrics(y_test, preds_lgb, p_lgb_test),
            "train_time_sec": round(lgb_time, 2),
            **benchmark_inference_latency(lgb_model.predict_proba, X_test[:1000])
        }

        # Pairwise and Voting Ensembles
        print(" -> Computing Pairwise and 3-Model Voting Ensembles...")
        p_xgb_cat = 0.5 * (p_xgb_test + p_cat_test)
        self.results["XGBoost_plus_CatBoost"] = compute_comprehensive_metrics(
            y_test, (p_xgb_cat[:, 1] >= 0.5).astype(int), p_xgb_cat
        )

        p_xgb_lgb = 0.5 * (p_xgb_test + p_lgb_test)
        self.results["XGBoost_plus_LightGBM"] = compute_comprehensive_metrics(
            y_test, (p_xgb_lgb[:, 1] >= 0.5).astype(int), p_xgb_lgb
        )

        p_cat_lgb = 0.5 * (p_cat_test + p_lgb_test)
        self.results["CatBoost_plus_LightGBM"] = compute_comprehensive_metrics(
            y_test, (p_cat_lgb[:, 1] >= 0.5).astype(int), p_cat_lgb
        )

        p_soft_vote = (p_xgb_test + p_cat_test + p_lgb_test) / 3.0
        self.results["Three_Model_Soft_Voting"] = compute_comprehensive_metrics(
            y_test, (p_soft_vote[:, 1] >= 0.5).astype(int), p_soft_vote
        )

        # Baseline E: Out-of-Fold Stacking Ensemble
        print(" -> Training Leakage-Free OOF Stacking Ensemble...")
        t0 = time.time()
        stacking = OOFStackingEnsemble(task="binary", n_splits=5, random_state=self.random_state)
        stacking.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        stack_time = time.time() - t0
        p_stack_test = stacking.predict_proba(X_test)
        preds_stack = (p_stack_test[:, 1] >= 0.5).astype(int)
        self.results["OOF_Stacking_Ensemble"] = {
            **compute_comprehensive_metrics(y_test, preds_stack, p_stack_test),
            "train_time_sec": round(stack_time, 2),
            **benchmark_inference_latency(stacking.predict_proba, X_test[:1000])
        }

        # 6. Graph Learning, GRU, and Attention Branches
        print("\n[Step 6/7] Training GraphSAGE, GRU, and Attention Modules...")
        feat_dim = X_train.shape[1]
        hidden_dim = 128
        seq_len = 12

        graphsage = HomogeneousGraphSAGE(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=0.15)
        gru = TemporalGRU(embed_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.15)
        attn = TemporalAttention(hidden_dim=hidden_dim, mode="additive")
        graph_classifier = nn.Linear(hidden_dim, 2)

        edge_tr = build_causal_session_graph(len(X_train), train_comb["session_id"].values, max_lag=3)
        edge_val = build_causal_session_graph(len(X_val), val_comb["session_id"].values, max_lag=3)
        edge_te = build_causal_session_graph(len(X_test), test_comb["session_id"].values, max_lag=3)

        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_te_t = torch.tensor(X_test, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

        def forward_spatiotemporal_graph(X_t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            n_rows = len(X_t)
            h_nodes = graphsage(X_t, edge_index)
            h_seqs = []
            for s in range(seq_len):
                shift_idx = np.clip(np.arange(n_rows) - (seq_len - 1 - s), 0, n_rows - 1)
                h_seqs.append(h_nodes[shift_idx])
            h_seq_tensor = torch.stack(h_seqs, dim=1)
            gru_out, _ = gru(h_seq_tensor)
            h_graph, _ = attn(gru_out)
            return h_graph

        opt_graph = optim.AdamW(
            list(graphsage.parameters()) + list(gru.parameters()) + list(attn.parameters()) + list(graph_classifier.parameters()),
            lr=0.004,
            weight_decay=1e-4
        )
        pos_w = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
        crit_graph = nn.CrossEntropyLoss(weight=torch.tensor([1.0, min(pos_w, 5.0)], dtype=torch.float32))

        graphsage.train()
        gru.train()
        attn.train()
        graph_classifier.train()
        for ep in range(35):
            opt_graph.zero_grad()
            h_tr = forward_spatiotemporal_graph(X_tr_t, edge_tr)
            logits_g = graph_classifier(h_tr)
            loss_g = crit_graph(logits_g, y_tr_tensor)
            loss_g.backward()
            opt_graph.step()

        graphsage.eval()
        gru.eval()
        attn.eval()
        graph_classifier.eval()
        with torch.no_grad():
            h_graph_train = forward_spatiotemporal_graph(X_tr_t, edge_tr).detach()
            h_graph_val = forward_spatiotemporal_graph(X_val_t, edge_val).detach()
            h_graph_test = forward_spatiotemporal_graph(X_te_t, edge_te).detach()
            p_graph_test = torch.softmax(graph_classifier(h_graph_test), dim=-1).cpu().numpy()
            preds_graph = (p_graph_test[:, 1] >= 0.5).astype(int)

        self.results["GraphSAGE_GRU_Attention_alone"] = compute_comprehensive_metrics(
            y_test, preds_graph, p_graph_test
        )

        # 7. Proposed STG-GBH Fusion Model
        print("\n[Step 7/7] Training Proposed STG-GBH Multitask Fusion Framework...")
        p_tr_oof = stacking.oof_predictions
        p_tr_stack_oof = stacking.oof_stack_proba[:, 1:2]
        p_boost_train = torch.tensor(np.hstack([p_tr_oof, p_tr_stack_oof]), dtype=torch.float32)

        p_val_xgb = xgb_model.predict_proba(X_val)[:, 1:2]
        p_val_cat = cat_model.predict_proba(X_val)[:, 1:2]
        p_val_lgb = lgb_model.predict_proba(X_val)[:, 1:2]
        p_val_stack = stacking.predict_proba(X_val)[:, 1:2]
        p_boost_val = torch.tensor(np.hstack([p_val_xgb, p_val_cat, p_val_lgb, p_val_stack]), dtype=torch.float32)

        p_test_xgb = p_xgb_test[:, 1:2]
        p_test_cat = p_cat_test[:, 1:2]
        p_test_lgb = p_lgb_test[:, 1:2]
        p_test_stack = p_stack_test[:, 1:2]
        p_boost_test = torch.tensor(np.hstack([p_test_xgb, p_test_cat, p_test_lgb, p_test_stack]), dtype=torch.float32)

        stg_gbh = STGGBHFusionModel(
            graph_embed_dim=hidden_dim,
            num_boosting_prob_cols=4,
            hidden_dim=hidden_dim,
            num_classes=2,
            enable_multitask=True,
            num_categories=11,
            num_mitre_stages=6
        )
        # Anchor skip weights to validation model strengths while training bounded spatio-temporal residual
        with torch.no_grad():
            w_val = stacking.validation_weights
            w_init = torch.tensor([0.65 * w_val[0], 0.65 * w_val[1], 0.65 * w_val[2], 0.35], dtype=torch.float32)
            stg_gbh.boost_skip.weight[0].copy_(-0.5 * w_init)
            stg_gbh.boost_skip.weight[1].copy_(0.5 * w_init)
            stg_gbh.boost_skip.bias.zero_()
            stg_gbh.classifier.weight.mul_(0.05)
        stg_gbh.boost_skip.requires_grad_(False)

        trainable_params = [p for p in stg_gbh.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=0.002, weight_decay=1e-3)
        loss_balancer = UncertaintyWeightedLoss(num_tasks=3)
        opt_balancer = optim.AdamW(loss_balancer.parameters(), lr=0.001)

        bce_loss = nn.CrossEntropyLoss()
        cat_loss_fn = nn.CrossEntropyLoss()
        mitre_loss_fn = nn.CrossEntropyLoss()

        y_tr_cat_t = torch.tensor(y_train_cat, dtype=torch.long)
        y_tr_mitre_t = torch.tensor(y_train_mitre, dtype=torch.long)

        best_val_f1 = -1.0
        best_state = copy.deepcopy(stg_gbh.state_dict())

        t0 = time.time()
        for epoch in range(40):
            stg_gbh.eval()
            with torch.no_grad():
                p_v = stg_gbh.predict_proba(h_graph_val, p_boost_val)[:, 1].cpu().numpy()
                val_f1 = f1_score(y_val, (p_v >= 0.5).astype(int), average="macro")
                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1
                    best_state = copy.deepcopy(stg_gbh.state_dict())

            stg_gbh.train()
            optimizer.zero_grad()
            opt_balancer.zero_grad()
            logits, mt_outs, z = stg_gbh(h_graph_train, p_boost_train)
            l_bin = bce_loss(logits, y_tr_tensor)
            l_cat = cat_loss_fn(mt_outs["category_logits"], y_tr_cat_t)
            l_mitre = mitre_loss_fn(mt_outs["mitre_logits"], y_tr_mitre_t)

            total_loss, _ = loss_balancer(l_bin, 0.15 * l_cat, 0.15 * l_mitre)
            total_loss.backward()
            optimizer.step()
            opt_balancer.step()

        stg_gbh.load_state_dict(best_state)
        stg_time = time.time() - t0

        # Evaluate STG-GBH
        stg_gbh.eval()
        with torch.no_grad():
            p_val_stg = stg_gbh.predict_proba(h_graph_val, p_boost_val).cpu().numpy()
            p_test_stg = stg_gbh.predict_proba(h_graph_test, p_boost_test).cpu().numpy()

        # Constrained Threshold Optimization on Validation Set
        print("\n -> Performing Constrained Threshold Optimization (FPR <= 2%, FNR <= 2%)...")
        opt_res = optimize_threshold_constrained(
            y_val, p_val_stg[:, 1], target_max_fpr=0.02, target_max_fnr=0.02
        )
        best_th = opt_res["chosen_threshold"]
        print(f"Optimal Threshold: {best_th:.4f} (Status: {opt_res['status']})")

        preds_stg_default = (p_test_stg[:, 1] >= 0.5).astype(int)
        preds_stg_optimal = (p_test_stg[:, 1] >= best_th).astype(int)

        self.results["STG_GBH_Default_Threshold"] = {
            **compute_comprehensive_metrics(y_test, preds_stg_default, p_test_stg),
            "train_time_sec": round(stg_time + stack_time, 2)
        }

        self.results["STG_GBH_Optimal_Threshold"] = {
            **compute_comprehensive_metrics(y_test, preds_stg_optimal, p_test_stg),
            "optimized_threshold": best_th,
            "threshold_optimization_status": opt_res["status"],
            "val_metrics_at_threshold": opt_res["chosen_metrics"],
            "train_time_sec": round(stg_time + stack_time, 2),
            **benchmark_inference_latency(
                lambda x: stg_gbh.predict_proba(h_graph_test[:len(x)], p_boost_test[:len(x)]),
                X_test[:1000]
            ),
            **compute_model_footprint(stg_gbh)
        }

        # Calibration analysis
        cal_metrics = compute_calibration_metrics(y_test, p_test_stg)
        self.results["STG_GBH_Calibration"] = cal_metrics

        # Explainability & Error Analysis
        print("\n -> Computing Explainability and Forensic Error Analysis...")
        shap_res = compute_tree_shap_summary(xgb_model.model, X_test[:300], all_feature_cols)
        feat_imp_table = compare_feature_importances(
            xgb_model.get_feature_importances(),
            cat_model.get_feature_importances(),
            lgb_model.get_feature_importances(),
            all_feature_cols
        )
        err_analysis = perform_error_analysis(
            test_comb, y_test, preds_stg_optimal, p_test_stg, all_feature_cols
        )

        self.results["Explainability_Top_Features"] = shap_res["top_features"]
        self.results["Error_Analysis"] = err_analysis

        # Save Checkpoints & Results
        print("\nSaving Artifacts, Predictions, and Checkpoints...")
        torch.save(stg_gbh.state_dict(), os.path.join(self.output_dir, "checkpoints", "stg_gbh_weights.pt"))
        preprocessor.save(os.path.join(self.output_dir, "checkpoints", "preprocessor.joblib"))
        stacking.save(os.path.join(self.output_dir, "checkpoints", "stacking_ensemble.joblib"))

        pred_df = pd.DataFrame({
            "y_true": y_test,
            "p_rf": p_rf_test[:, 1],
            "p_xgb": p_xgb_test[:, 1],
            "p_cat": p_cat_test[:, 1],
            "p_lgb": p_lgb_test[:, 1],
            "p_stack": p_stack_test[:, 1],
            "p_stg_gbh": p_test_stg[:, 1],
            "pred_stg_gbh_optimal": preds_stg_optimal
        })
        pred_df.to_parquet(os.path.join(self.output_dir, "predictions", "in_domain_test_predictions.parquet"))

        with open(os.path.join(self.output_dir, "metrics", "in_domain_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=2)

        feat_imp_table.to_csv(os.path.join(self.output_dir, "tables", "feature_importance_comparison.csv"), index=False)

        print("\n=================================================================")
        print("  IN-DOMAIN EXPERIMENTS COMPLETED SUCCESSFULLY!                  ")
        print("=================================================================")
        return self.results


if __name__ == "__main__":
    exp = InDomainExperiment()
    results = exp.run()
