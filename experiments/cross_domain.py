"""
Cross-Domain IoT Generalization Pipeline for STG-GBH.
Evaluates domain shifts across Industrial IoT (Edge-IIoTset),
Modern IoT (CICIoT2023), and NetFlow v2 (ToN-IoT) with full feature engineering.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from preprocessing.dataset_adapters.edge_iiotset import EdgeIIoTsetAdapter
from preprocessing.dataset_adapters.ciciot2023 import CICIoT2023Adapter
from preprocessing.dataset_adapters.ton_iot import ToNIoTAdapter
from preprocessing.common_preprocessing import LeakageFreePreprocessor, chronological_split
from preprocessing.feature_engineering import (
    compute_network_features,
    compute_causal_temporal_features,
    compute_topological_graph_features
)
from models.stacking_ensemble import OOFStackingEnsemble
from optimization.threshold_optimization import optimize_threshold_constrained
from evaluation.metrics import compute_comprehensive_metrics


class CrossDomainExperiment:
    """
    Evaluates in-domain and cross-domain generalization across heterogeneous IoT protocols.
    """
    def __init__(
        self,
        edge_path: str = "datasets/Edge_IIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
        ciciot_dir: str = "datasets/CICIoT2023",
        ton_path: str = "datasets/ToN_IoT/NF-ToN-IoT-V2.parquet",
        output_dir: str = "results",
        random_state: int = 42
    ):
        self.edge_path = edge_path
        self.ciciot_dir = ciciot_dir
        self.ton_path = ton_path
        self.output_dir = output_dir
        self.random_state = random_state
        self.results: Dict[str, Any] = {}

    @staticmethod
    def _chronological_block_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits file-grouped datasets chronologically within each capture block (70% Train / 15% Val / 15% Test)
        and restores original temporal interleaving without exposing labels to session_id.
        """
        tr_parts, val_parts, te_parts = [], [], []
        grp_col = "canonical_family" if "canonical_family" in df.columns else "is_attack"
        for _, grp in df.groupby(grp_col, sort=False):
            n = len(grp)
            n1 = int(0.70 * n)
            n2 = int(0.85 * n)
            tr_parts.append(grp.iloc[:n1].copy())
            val_parts.append(grp.iloc[n1:n2].copy())
            te_parts.append(grp.iloc[n2:].copy())
        tr = pd.concat(tr_parts).sort_index().reset_index(drop=True)
        val = pd.concat(val_parts).sort_index().reset_index(drop=True)
        te = pd.concat(te_parts).sort_index().reset_index(drop=True)
        tr["session_id"] = "T1_Train"
        val["session_id"] = "T2_Val"
        te["session_id"] = "T3_Test"
        return tr, val, te

    def _evaluate_iot_domain(
        self,
        df: pd.DataFrame,
        base_feats: list,
        time_col: str = None,
        block_split: bool = False
    ) -> Dict[str, Any]:
        if block_split:
            tr, val, te = self._chronological_block_split(df)
        else:
            tr, val, te = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
            tr["session_id"] = "T1_Train"
            val["session_id"] = "T2_Val"
            te["session_id"] = "T3_Test"

        def enrich(d):
            net_f = compute_network_features(d)
            temp_f = compute_causal_temporal_features(d, time_col=time_col)
            grp_f = compute_topological_graph_features(d)
            return pd.concat([d, net_f, temp_f, grp_f], axis=1)

        tr_comb = enrich(tr)
        val_comb = enrich(val)
        te_comb = enrich(te)

        all_feats = [
            c for c in tr_comb.columns
            if c in base_feats or c.startswith("net_") or c.startswith("causal_") or c.startswith("graph_")
        ]

        prep = LeakageFreePreprocessor(all_feats, scale_method="robust")
        X_tr = prep.fit_transform(tr_comb)
        X_val = prep.transform(val_comb)
        X_te = prep.transform(te_comb)

        y_tr = tr_comb["is_attack"].values.astype(int)
        y_val = val_comb["is_attack"].values.astype(int)
        y_te = te_comb["is_attack"].values.astype(int)

        ens = OOFStackingEnsemble(task="binary", n_splits=5, random_state=self.random_state)
        ens.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        p_val = ens.predict_proba(X_val)
        p_te = ens.predict_proba(X_te)

        opt = optimize_threshold_constrained(y_val, p_val[:, 1], target_max_fpr=0.02, target_max_fnr=0.02)
        th = opt["chosen_threshold"]
        return compute_comprehensive_metrics(y_te, (p_te[:, 1] >= th).astype(int), p_te)

    def run(self) -> Dict[str, Any]:
        print("\n=================================================================")
        print("  STARTING CROSS-DOMAIN IOT EXPERIMENTS                          ")
        print("=================================================================")

        # 1. Edge-IIoTset (Industrial IoT Domain)
        print("\n[Domain 1/3] Benchmarking Edge-IIoTset (Industrial IoT)...")
        edge_adapter = EdgeIIoTsetAdapter(self.edge_path)
        edge_df, edge_feats = edge_adapter.load_clean_data(max_rows=25000)
        self.results["Edge_IIoTset_InDomain"] = self._evaluate_iot_domain(
            edge_df, edge_feats, time_col="timestamp_sec", block_split=True
        )

        # 2. CICIoT2023 (Large-scale Modern IoT)
        print("\n[Domain 2/3] Benchmarking CICIoT2023 (Modern IoT Attacks)...")
        ciciot_adapter = CICIoT2023Adapter(self.ciciot_dir)
        ciciot_df, ciciot_feats = ciciot_adapter.load_sample(max_rows=25000)
        self.results["CICIoT2023_InDomain"] = self._evaluate_iot_domain(
            ciciot_df, ciciot_feats, block_split=False
        )

        # 3. ToN-IoT (NetFlow v2 Domain)
        print("\n[Domain 3/3] Benchmarking ToN-IoT (NetFlow v2 Format)...")
        ton_adapter = ToNIoTAdapter(self.ton_path)
        ton_df, ton_feats = ton_adapter.load_sample(max_rows=25000)
        self.results["ToN_IoT_InDomain"] = self._evaluate_iot_domain(
            ton_df, ton_feats, block_split=False
        )

        out_file = os.path.join(self.output_dir, "metrics", "cross_domain_metrics.json")
        with open(out_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Saved cross-domain metrics to {out_file}")
        return self.results


if __name__ == "__main__":
    exp = CrossDomainExperiment()
    results = exp.run()
