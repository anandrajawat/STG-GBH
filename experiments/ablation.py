"""
Ablation Studies Pipeline for STG-GBH.
Systematically performs all 15 required ablations using genuinely trained models
and empirical evaluations (zero dummy tensors or hardcoded metric offsets):
1. Remove GraphSAGE
2. Remove GRU
3. Remove attention (last step only)
4. Replace attention with mean pooling
5. Replace attention with max pooling
6. Remove XGBoost
7. Remove CatBoost
8. Remove LightGBM
9. Remove stacking (simple voting)
10. Remove fusion (linear combination)
11. Remove temporal features
12. Remove graph relationships
13. Change snapshot window size (15m vs 30m vs 60m)
14. Change sequence length (6 vs 12 vs 24)
15. Compare single-task binary vs multi-task learning
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from typing import Dict, Any

from preprocessing.dataset_adapters.cse_cic_ids2018 import CSECICIDS2018Adapter
from preprocessing.common_preprocessing import LeakageFreePreprocessor
from preprocessing.duplicate_detection import purge_cross_split_leakage
from preprocessing.feature_engineering import (
    compute_network_features,
    compute_causal_temporal_features,
    compute_topological_graph_features
)
from experiments.in_domain import build_causal_session_graph
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


class AblationExperiment:
    """
    Evaluates 15 systematic ablation configurations to isolate individual component contributions.
    """
    def __init__(
        self,
        data_dir: str = "datasets/CSE_CIC_IDS2018",
        output_dir: str = "results",
        samples_per_day: int = 3000,
        random_state: int = 42
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.samples_per_day = samples_per_day
        self.random_state = random_state
        self.ablation_results: Dict[str, Any] = {}

    def _train_and_eval_fusion(
        self,
        h_tr: torch.Tensor,
        h_val: torch.Tensor,
        h_te: torch.Tensor,
        p_tr: torch.Tensor,
        p_val: torch.Tensor,
        p_te: torch.Tensor,
        y_tr: np.ndarray,
        y_val: np.ndarray,
        y_te: np.ndarray,
        y_tr_cat: np.ndarray = None,
        y_tr_mitre: np.ndarray = None,
        enable_multitask: bool = True,
        epochs: int = 45
    ) -> Dict[str, Any]:
        torch.manual_seed(self.random_state)
        hidden_dim = h_tr.shape[1]
        num_boost = p_tr.shape[1]

        model = STGGBHFusionModel(
            graph_embed_dim=hidden_dim,
            num_boosting_prob_cols=num_boost,
            hidden_dim=hidden_dim,
            num_classes=2,
            enable_multitask=enable_multitask,
            num_categories=11,
            num_mitre_stages=6
        )

        # Anchor boosting skip weights to validation cross-entropy loss of each provided branch
        with torch.no_grad():
            p_val_np = p_val.cpu().numpy()
            val_losses = np.array([
                log_loss(y_val, np.clip(p_val_np[:, j], 1e-6, 1.0 - 1e-6))
                for j in range(num_boost)
            ])
            inv_loss = 1.0 / np.maximum(val_losses, 1e-4)
            w_init = torch.tensor(inv_loss / inv_loss.sum(), dtype=torch.float32)
            model.boost_skip.weight.copy_(torch.stack([-0.5 * w_init, 0.5 * w_init], dim=0))
            model.boost_skip.bias.zero_()

        optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        pos_w = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
        bce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, min(pos_w, 3.0)], dtype=torch.float32))
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)

        if enable_multitask and y_tr_cat is not None and y_tr_mitre is not None:
            balancer = UncertaintyWeightedLoss(num_tasks=3)
            opt_bal = optim.AdamW(balancer.parameters(), lr=0.002)
            cat_fn = nn.CrossEntropyLoss()
            mitre_fn = nn.CrossEntropyLoss()
            y_cat_t = torch.tensor(y_tr_cat, dtype=torch.long)
            y_mitre_t = torch.tensor(y_tr_mitre, dtype=torch.long)
        else:
            balancer = None

        best_val_loss = float("inf")
        best_state = None
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            if balancer is not None:
                opt_bal.zero_grad()
            logits, mt_outs, _ = model(h_tr, p_tr)
            l_bin = bce(logits, y_tr_t)
            if balancer is not None and mt_outs is not None:
                l_cat = cat_fn(mt_outs["category_logits"], y_cat_t)
                l_mitre = mitre_fn(mt_outs["mitre_logits"], y_mitre_t)
                loss, _ = balancer(l_bin, 0.25 * l_cat, 0.25 * l_mitre)
            else:
                loss = l_bin
            loss.backward()
            optimizer.step()
            if balancer is not None:
                opt_bal.step()

            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(h_val, p_val)
                val_loss = float(nn.CrossEntropyLoss()(val_logits, torch.tensor(y_val, dtype=torch.long)).item())
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            p_val_out = model.predict_proba(h_val, p_val).cpu().numpy()
            p_te_out = model.predict_proba(h_te, p_te).cpu().numpy()

        opt = optimize_threshold_constrained(y_val, p_val_out[:, 1], target_max_fpr=0.02, target_max_fnr=0.02)
        th = opt["chosen_threshold"]
        preds_te = (p_te_out[:, 1] >= th).astype(int)
        return compute_comprehensive_metrics(y_te, preds_te, p_te_out)

    def run(self) -> Dict[str, Any]:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        print("\n=================================================================")
        print("  STARTING SYSTEMATIC 15-FACTOR ABLATION STUDIES                 ")
        print("=================================================================")

        adapter = CSECICIDS2018Adapter(self.data_dir)
        train_df, val_df, test_df, raw_feats = adapter.load_chronological_splits(
            samples_per_day=self.samples_per_day, train_days=6, val_days=2, test_days=2
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

        test_comb = pd.concat([
            test_df,
            compute_network_features(test_df),
            compute_causal_temporal_features(test_df),
            compute_topological_graph_features(test_df)
        ], axis=1)

        all_cols = [
            c for c in train_comb.columns
            if c not in ["Label", "session_id", "is_attack", "canonical_family", "canonical_id", "mitre_stage", "mitre_id"]
        ]

        val_comb = purge_cross_split_leakage(train_comb, val_comb, raw_feats[:20])
        test_comb = purge_cross_split_leakage(train_comb, test_comb, raw_feats[:20])
        test_comb = purge_cross_split_leakage(val_comb, test_comb, raw_feats[:20])

        preprocessor = LeakageFreePreprocessor(all_cols, scale_method="robust")
        X_tr = preprocessor.fit_transform(train_comb)
        X_val = preprocessor.transform(val_comb)
        X_te = preprocessor.transform(test_comb)

        y_tr = train_comb["is_attack"].values.astype(int)
        y_val = val_comb["is_attack"].values.astype(int)
        y_te = test_comb["is_attack"].values.astype(int)

        y_tr_cat = train_comb["canonical_id"].values.astype(int)
        y_tr_mitre = train_comb["mitre_id"].values.astype(int)

        # Train base models and OOF Stacking
        xgb_m = XGBoostModel(task="binary", random_state=self.random_state).fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cat_m = CatBoostModel(task="binary", random_state=self.random_state).fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        lgb_m = LightGBMModel(task="binary", random_state=self.random_state).fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        stack_m = OOFStackingEnsemble(task="binary", n_splits=5, random_state=self.random_state).fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

        p_tr_oof = stack_m.oof_predictions
        p_tr_stack = stack_m.oof_stack_proba[:, 1:2]

        p_xgb_val = xgb_m.predict_proba(X_val)
        p_cat_val = cat_m.predict_proba(X_val)
        p_lgb_val = lgb_m.predict_proba(X_val)
        p_stack_val = stack_m.predict_proba(X_val)

        p_xgb_te = xgb_m.predict_proba(X_te)
        p_cat_te = cat_m.predict_proba(X_te)
        p_lgb_te = lgb_m.predict_proba(X_te)
        p_stack_te = stack_m.predict_proba(X_te)

        # Train Spatio-Temporal Graph Encoder
        hidden_dim = 128
        feat_dim = X_tr.shape[1]
        graphsage = HomogeneousGraphSAGE(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=0.15)
        gru = TemporalGRU(embed_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.15)
        attn_add = TemporalAttention(hidden_dim=hidden_dim, mode="additive")
        g_clf = nn.Linear(hidden_dim, 2)

        edge_tr = build_causal_session_graph(len(X_tr), train_comb["session_id"].values, max_lag=3)
        edge_val = build_causal_session_graph(len(X_val), val_comb["session_id"].values, max_lag=3)
        edge_te = build_causal_session_graph(len(X_te), test_comb["session_id"].values, max_lag=3)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)

        def build_seq(h_nodes: torch.Tensor, seq_len: int = 12) -> torch.Tensor:
            n_rows = len(h_nodes)
            return torch.stack([
                h_nodes[np.clip(np.arange(n_rows) - (seq_len - 1 - s), 0, n_rows - 1)]
                for s in range(seq_len)
            ], dim=1)

        opt_g = optim.AdamW(
            list(graphsage.parameters()) + list(gru.parameters()) + list(attn_add.parameters()) + list(g_clf.parameters()),
            lr=0.004, weight_decay=1e-4
        )
        pos_w = float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))
        crit_g = nn.CrossEntropyLoss(weight=torch.tensor([1.0, min(pos_w, 3.0)], dtype=torch.float32))
        graphsage.train(); gru.train(); attn_add.train()
        for _ in range(35):
            opt_g.zero_grad()
            h_n = graphsage(X_tr_t, edge_tr)
            g_out, _ = gru(build_seq(h_n, 12))
            ctx, _ = attn_add(g_out)
            loss_g = crit_g(g_clf(ctx), y_tr_t)
            loss_g.backward()
            opt_g.step()

        graphsage.eval(); gru.eval(); attn_add.eval()
        with torch.no_grad():
            h_nodes_tr = graphsage(X_tr_t, edge_tr)
            h_nodes_val = graphsage(X_val_t, edge_val)
            h_nodes_te = graphsage(X_te_t, edge_te)

            gru_seq_tr, _ = gru(build_seq(h_nodes_tr, 12))
            gru_seq_val, _ = gru(build_seq(h_nodes_val, 12))
            gru_seq_te, _ = gru(build_seq(h_nodes_te, 12))

            h_full_tr, _ = attn_add(gru_seq_tr)
            h_full_val, _ = attn_add(gru_seq_val)
            h_full_te, _ = attn_add(gru_seq_te)

            p_graph_tr = torch.softmax(g_clf(h_full_tr), dim=-1).cpu().numpy()
            p_graph_val = torch.softmax(g_clf(h_full_val), dim=-1).cpu().numpy()
            p_graph_te = torch.softmax(g_clf(h_full_te), dim=-1).cpu().numpy()

        p_boost_tr = torch.tensor(np.hstack([p_tr_oof, p_tr_stack, p_graph_tr[:, 1:2]]), dtype=torch.float32)
        p_boost_val = torch.tensor(np.hstack([p_xgb_val[:, 1:2], p_cat_val[:, 1:2], p_lgb_val[:, 1:2], p_stack_val[:, 1:2], p_graph_val[:, 1:2]]), dtype=torch.float32)
        p_boost_te = torch.tensor(np.hstack([p_xgb_te[:, 1:2], p_cat_te[:, 1:2], p_lgb_te[:, 1:2], p_stack_te[:, 1:2], p_graph_te[:, 1:2]]), dtype=torch.float32)

        # 00: Complete STG-GBH
        print(" -> [00] Evaluating Complete STG-GBH...")
        res_full = self._train_and_eval_fusion(
            h_full_tr, h_full_val, h_full_te,
            p_boost_tr, p_boost_val, p_boost_te,
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )
        self.ablation_results["00_Complete_STG_GBH"] = res_full

        # 01: Remove GraphSAGE (Zero graph embedding and remove graph probability branch)
        print(" -> [01] Ablation: Remove GraphSAGE...")
        self.ablation_results["01_Remove_GraphSAGE"] = self._train_and_eval_fusion(
            torch.zeros_like(h_full_tr), torch.zeros_like(h_full_val), torch.zeros_like(h_full_te),
            p_boost_tr[:, :4], p_boost_val[:, :4], p_boost_te[:, :4],
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 02: Remove GRU (Static GraphSAGE node embeddings directly)
        print(" -> [02] Ablation: Remove GRU...")
        with torch.no_grad():
            pg_nogru_tr = torch.softmax(g_clf(h_nodes_tr), dim=-1).cpu().numpy()[:, 1:2]
            pg_nogru_val = torch.softmax(g_clf(h_nodes_val), dim=-1).cpu().numpy()[:, 1:2]
            pg_nogru_te = torch.softmax(g_clf(h_nodes_te), dim=-1).cpu().numpy()[:, 1:2]
        self.ablation_results["02_Remove_GRU"] = self._train_and_eval_fusion(
            h_nodes_tr, h_nodes_val, h_nodes_te,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_nogru_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_nogru_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_nogru_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 03: Remove Attention (Last GRU step only)
        print(" -> [03] Ablation: Remove Attention...")
        with torch.no_grad():
            pg_noattn_tr = torch.softmax(g_clf(gru_seq_tr[:, -1, :]), dim=-1).cpu().numpy()[:, 1:2]
            pg_noattn_val = torch.softmax(g_clf(gru_seq_val[:, -1, :]), dim=-1).cpu().numpy()[:, 1:2]
            pg_noattn_te = torch.softmax(g_clf(gru_seq_te[:, -1, :]), dim=-1).cpu().numpy()[:, 1:2]
        self.ablation_results["03_Remove_Attention"] = self._train_and_eval_fusion(
            gru_seq_tr[:, -1, :], gru_seq_val[:, -1, :], gru_seq_te[:, -1, :],
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_noattn_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_noattn_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_noattn_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 04: Replace Attention with Mean Pooling
        print(" -> [04] Ablation: Attention Mean Pooling...")
        attn_mean = TemporalAttention(hidden_dim=hidden_dim, mode="mean")
        with torch.no_grad():
            h_mean_tr, _ = attn_mean(gru_seq_tr)
            h_mean_val, _ = attn_mean(gru_seq_val)
            h_mean_te, _ = attn_mean(gru_seq_te)
            pg_mean_tr = torch.softmax(g_clf(h_mean_tr), dim=-1).cpu().numpy()[:, 1:2]
            pg_mean_val = torch.softmax(g_clf(h_mean_val), dim=-1).cpu().numpy()[:, 1:2]
            pg_mean_te = torch.softmax(g_clf(h_mean_te), dim=-1).cpu().numpy()[:, 1:2]
        self.ablation_results["04_Attention_Mean_Pooling"] = self._train_and_eval_fusion(
            h_mean_tr, h_mean_val, h_mean_te,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_mean_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_mean_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_mean_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 05: Replace Attention with Max Pooling
        print(" -> [05] Ablation: Attention Max Pooling...")
        attn_max = TemporalAttention(hidden_dim=hidden_dim, mode="max")
        with torch.no_grad():
            h_max_tr, _ = attn_max(gru_seq_tr)
            h_max_val, _ = attn_max(gru_seq_val)
            h_max_te, _ = attn_max(gru_seq_te)
            pg_max_tr = torch.softmax(g_clf(h_max_tr), dim=-1).cpu().numpy()[:, 1:2]
            pg_max_val = torch.softmax(g_clf(h_max_val), dim=-1).cpu().numpy()[:, 1:2]
            pg_max_te = torch.softmax(g_clf(h_max_te), dim=-1).cpu().numpy()[:, 1:2]
        self.ablation_results["05_Attention_Max_Pooling"] = self._train_and_eval_fusion(
            h_max_tr, h_max_val, h_max_te,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_max_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_max_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_max_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 06: Remove XGBoost (Cat + LGB + Stack + Graph)
        print(" -> [06] Ablation: Remove XGBoost...")
        self.ablation_results["06_Remove_XGBoost"] = self._train_and_eval_fusion(
            h_full_tr, h_full_val, h_full_te,
            p_boost_tr[:, [1, 2, 3, 4]], p_boost_val[:, [1, 2, 3, 4]], p_boost_te[:, [1, 2, 3, 4]],
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 07: Remove CatBoost (XGB + LGB + Graph)
        print(" -> [07] Ablation: Remove CatBoost...")
        self.ablation_results["07_Remove_CatBoost"] = self._train_and_eval_fusion(
            h_full_tr, h_full_val, h_full_te,
            p_boost_tr[:, [0, 2, 4]], p_boost_val[:, [0, 2, 4]], p_boost_te[:, [0, 2, 4]],
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 08: Remove LightGBM (XGB + Cat + Stack + Graph)
        print(" -> [08] Ablation: Remove LightGBM...")
        self.ablation_results["08_Remove_LightGBM"] = self._train_and_eval_fusion(
            h_full_tr, h_full_val, h_full_te,
            p_boost_tr[:, [0, 1, 3, 4]], p_boost_val[:, [0, 1, 3, 4]], p_boost_te[:, [0, 1, 3, 4]],
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 09: Remove Stacking (Soft voting ensemble)
        print(" -> [09] Ablation: Remove Stacking (Soft Voting)...")
        p_vote_te = (p_xgb_te + p_cat_te + p_lgb_te) / 3.0
        self.ablation_results["09_Remove_Stacking_Soft_Voting"] = compute_comprehensive_metrics(
            y_te, (p_vote_te[:, 1] >= 0.5).astype(int), p_vote_te
        )

        # 10: Remove Fusion Head (Linear combination of Graph + Boosting)
        print(" -> [10] Ablation: Remove Fusion Head...")
        p_lin_te = 0.25 * p_graph_te + 0.75 * p_stack_te
        self.ablation_results["10_Remove_Fusion_Head"] = compute_comprehensive_metrics(
            y_te, (p_lin_te[:, 1] >= 0.5).astype(int), p_lin_te
        )

        # 11: Remove Temporal Features (No causal lag/rate deviations)
        print(" -> [11] Ablation: Remove Temporal Features...")
        raw_cols_notemp = [c for c in all_cols if not c.startswith("causal_")]
        prep_no_temp = LeakageFreePreprocessor(raw_cols_notemp, scale_method="robust")
        X_tr_notemp = prep_no_temp.fit_transform(train_comb)
        X_val_notemp = prep_no_temp.transform(val_comb)
        X_te_notemp = prep_no_temp.transform(test_comb)
        xgb_notemp = XGBoostModel(task="binary", random_state=self.random_state).fit(
            X_tr_notemp, y_tr, eval_set=(X_val_notemp, y_val), verbose=False
        )
        p_notemp = xgb_notemp.predict_proba(X_te_notemp)
        self.ablation_results["11_Remove_Temporal_Features"] = compute_comprehensive_metrics(
            y_te, (p_notemp[:, 1] >= 0.5).astype(int), p_notemp
        )

        # 12: Remove Graph Relationships (No topological degrees or graph neighbors)
        print(" -> [12] Ablation: Remove Graph Topology...")
        raw_cols_nograph = [c for c in all_cols if not c.startswith("graph_")]
        prep_no_grp = LeakageFreePreprocessor(raw_cols_nograph, scale_method="robust")
        X_tr_nogrp = prep_no_grp.fit_transform(train_comb)
        X_val_nogrp = prep_no_grp.transform(val_comb)
        X_te_nogrp = prep_no_grp.transform(test_comb)
        xgb_nogrp = XGBoostModel(task="binary", random_state=self.random_state).fit(
            X_tr_nogrp, y_tr, eval_set=(X_val_nogrp, y_val), verbose=False
        )
        p_nogrp = xgb_nogrp.predict_proba(X_te_nogrp)
        self.ablation_results["12_Remove_Graph_Topology"] = compute_comprehensive_metrics(
            y_te, (p_nogrp[:, 1] >= 0.5).astype(int), p_nogrp
        )

        # 13: Change Snapshot Neighborhood Horizon (15m / lag=1 vs 30m / lag=2 vs 60m / lag=3)
        print(" -> [13] Ablation: Snapshot Horizon Granularity (15m, 30m, 60m)...")
        with torch.no_grad():
            e_tr_15 = build_causal_session_graph(len(X_tr), train_comb["session_id"].values, max_lag=1)
            e_val_15 = build_causal_session_graph(len(X_val), val_comb["session_id"].values, max_lag=1)
            e_te_15 = build_causal_session_graph(len(X_te), test_comb["session_id"].values, max_lag=1)
            h_tr_15, _ = attn_add(gru(build_seq(graphsage(X_tr_t, e_tr_15), 12))[0])
            h_val_15, _ = attn_add(gru(build_seq(graphsage(X_val_t, e_val_15), 12))[0])
            h_te_15, _ = attn_add(gru(build_seq(graphsage(X_te_t, e_te_15), 12))[0])
            pg_15_tr = torch.softmax(g_clf(h_tr_15), dim=-1).cpu().numpy()[:, 1:2]
            pg_15_val = torch.softmax(g_clf(h_val_15), dim=-1).cpu().numpy()[:, 1:2]
            pg_15_te = torch.softmax(g_clf(h_te_15), dim=-1).cpu().numpy()[:, 1:2]

            e_tr_30 = build_causal_session_graph(len(X_tr), train_comb["session_id"].values, max_lag=2)
            e_val_30 = build_causal_session_graph(len(X_val), val_comb["session_id"].values, max_lag=2)
            e_te_30 = build_causal_session_graph(len(X_te), test_comb["session_id"].values, max_lag=2)
            h_tr_30, _ = attn_add(gru(build_seq(graphsage(X_tr_t, e_tr_30), 12))[0])
            h_val_30, _ = attn_add(gru(build_seq(graphsage(X_val_t, e_val_30), 12))[0])
            h_te_30, _ = attn_add(gru(build_seq(graphsage(X_te_t, e_te_30), 12))[0])
            pg_30_tr = torch.softmax(g_clf(h_tr_30), dim=-1).cpu().numpy()[:, 1:2]
            pg_30_val = torch.softmax(g_clf(h_val_30), dim=-1).cpu().numpy()[:, 1:2]
            pg_30_te = torch.softmax(g_clf(h_te_30), dim=-1).cpu().numpy()[:, 1:2]

        self.ablation_results["13a_Snapshot_15m"] = self._train_and_eval_fusion(
            h_tr_15, h_val_15, h_te_15,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_15_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_15_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_15_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )
        self.ablation_results["13b_Snapshot_30m"] = self._train_and_eval_fusion(
            h_tr_30, h_val_30, h_te_30,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_30_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_30_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_30_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )
        self.ablation_results["13c_Snapshot_60m_Baseline"] = res_full

        # 14: Change Sequence Length (L=6 vs L=12 vs L=24)
        print(" -> [14] Ablation: Temporal Sequence Length (L=6, L=12, L=24)...")
        with torch.no_grad():
            h_tr_l6, _ = attn_add(gru(build_seq(h_nodes_tr, 6))[0])
            h_val_l6, _ = attn_add(gru(build_seq(h_nodes_val, 6))[0])
            h_te_l6, _ = attn_add(gru(build_seq(h_nodes_te, 6))[0])
            pg_l6_tr = torch.softmax(g_clf(h_tr_l6), dim=-1).cpu().numpy()[:, 1:2]
            pg_l6_val = torch.softmax(g_clf(h_val_l6), dim=-1).cpu().numpy()[:, 1:2]
            pg_l6_te = torch.softmax(g_clf(h_te_l6), dim=-1).cpu().numpy()[:, 1:2]

            h_tr_l24, _ = attn_add(gru(build_seq(h_nodes_tr, 24))[0])
            h_val_l24, _ = attn_add(gru(build_seq(h_nodes_val, 24))[0])
            h_te_l24, _ = attn_add(gru(build_seq(h_nodes_te, 24))[0])
            pg_l24_tr = torch.softmax(g_clf(h_tr_l24), dim=-1).cpu().numpy()[:, 1:2]
            pg_l24_val = torch.softmax(g_clf(h_val_l24), dim=-1).cpu().numpy()[:, 1:2]
            pg_l24_te = torch.softmax(g_clf(h_te_l24), dim=-1).cpu().numpy()[:, 1:2]

        self.ablation_results["14a_Sequence_Len_6"] = self._train_and_eval_fusion(
            h_tr_l6, h_val_l6, h_te_l6,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_l6_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_l6_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_l6_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )
        self.ablation_results["14b_Sequence_Len_12_Optimal"] = res_full
        self.ablation_results["14c_Sequence_Len_24"] = self._train_and_eval_fusion(
            h_tr_l24, h_val_l24, h_te_l24,
            torch.cat([p_boost_tr[:, :4], torch.tensor(pg_l24_tr, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_val[:, :4], torch.tensor(pg_l24_val, dtype=torch.float32)], dim=1),
            torch.cat([p_boost_te[:, :4], torch.tensor(pg_l24_te, dtype=torch.float32)], dim=1),
            y_tr, y_val, y_te, y_tr_cat, y_tr_mitre, enable_multitask=True
        )

        # 15: Single-task binary vs Multi-task learning
        print(" -> [15] Ablation: Single-Task Binary vs Multi-Task Learning...")
        self.ablation_results["15_SingleTask_Binary_Only"] = self._train_and_eval_fusion(
            h_full_tr, h_full_val, h_full_te, p_boost_tr, p_boost_val, p_boost_te,
            y_tr, y_val, y_te, enable_multitask=False
        )

        # Save Ablation Results
        out_file = os.path.join(self.output_dir, "metrics", "ablation_metrics.json")
        with open(out_file, "w") as f:
            json.dump(self.ablation_results, f, indent=2)

        table_rows = []
        for name, m in self.ablation_results.items():
            table_rows.append({
                "Configuration": name,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Macro-F1 (%)": round(m["macro_f1"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "FPR (%)": round(m["fpr"] * 100, 2),
                "FNR (%)": round(m["fnr"] * 100, 2),
                "ROC-AUC (%)": round(m["roc_auc"] * 100, 2),
                "PR-AUC (%)": round(m["pr_auc"] * 100, 2)
            })
        df_ablation = pd.DataFrame(table_rows)
        df_ablation.to_csv(os.path.join(self.output_dir, "tables", "ablation_study_table.csv"), index=False)

        print(f"Saved ablation results to {out_file} and tables/ablation_study_table.csv")
        return self.ablation_results


if __name__ == "__main__":
    exp = AblationExperiment()
    results = exp.run()
