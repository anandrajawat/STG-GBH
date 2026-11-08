"""
Publication-ready visualizer and table generator for STG-GBH.
Generates 300 DPI publication plots (Matplotlib/Seaborn) and academic tables.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

# Set academic typography and style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def generate_all_publication_artifacts(results_dir: str = "results"):
    figs_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    metrics_file = os.path.join(results_dir, "metrics", "in_domain_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found. Run experiments first.")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    pred_file = os.path.join(results_dir, "predictions", "in_domain_test_predictions.parquet")
    preds_df = pd.read_parquet(pred_file) if os.path.exists(pred_file) else None

    # 1. Model Comparison Table
    rows = []
    for model_name, m in metrics.items():
        if isinstance(m, dict) and "macro_f1" in m:
            rows.append({
                "Model Architecture": model_name.replace("_", " "),
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Macro-F1 (%)": round(m["macro_f1"] * 100, 2),
                "ROC-AUC (%)": round(m["roc_auc"] * 100, 2),
                "PR-AUC (%)": round(m["pr_auc"] * 100, 2),
                "FPR (%)": round(m["fpr"] * 100, 2),
                "FNR (%)": round(m["fnr"] * 100, 2),
                "Train Time (s)": m.get("train_time_sec", "N/A"),
                "Latency/1k (ms)": m.get("latency_per_1000_obs_ms", "N/A")
            })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(tables_dir, "model_comparison_table.csv"), index=False)
    print("Saved model_comparison_table.csv")

    # 2. Plot ROC Curves
    if preds_df is not None:
        plt.figure(figsize=(8, 6), dpi=300)
        y_true = preds_df["y_true"].values

        for col, label, color in [
            ("p_rf", "Random Forest", "#7f7f7f"),
            ("p_xgb", "XGBoost", "#1f77b4"),
            ("p_cat", "CatBoost", "#ff7f0e"),
            ("p_lgb", "LightGBM", "#2ca02c"),
            ("p_stack", "OOF Stacking", "#9467bd"),
            ("p_stg_gbh", "STG-GBH (Proposed)", "#d62728")
        ]:
            if col in preds_df.columns:
                fpr, tpr, _ = roc_curve(y_true, preds_df[col].values)
                auc_val = metrics.get(label.replace(" ", "_"), {}).get("roc_auc", 0.0)
                plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})", color=color, lw=2 if "STG" in label else 1.5)

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR / Recall)")
        plt.title("Receiver Operating Characteristic (ROC) Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "roc_curves.png"), dpi=300)
        plt.close()
        print("Saved roc_curves.png")

        # 3. Plot Precision-Recall Curves
        plt.figure(figsize=(8, 6), dpi=300)
        for col, label, color in [
            ("p_rf", "Random Forest", "#7f7f7f"),
            ("p_xgb", "XGBoost", "#1f77b4"),
            ("p_cat", "CatBoost", "#ff7f0e"),
            ("p_lgb", "LightGBM", "#2ca02c"),
            ("p_stack", "OOF Stacking", "#9467bd"),
            ("p_stg_gbh", "STG-GBH (Proposed)", "#d62728")
        ]:
            if col in preds_df.columns:
                prec, rec, _ = precision_recall_curve(y_true, preds_df[col].values)
                pr_auc = metrics.get(label.replace(" ", "_"), {}).get("pr_auc", 0.0)
                plt.plot(rec, prec, label=f"{label} (PR-AUC = {pr_auc:.3f})", color=color, lw=2 if "STG" in label else 1.5)

        plt.xlim([0.0, 1.02])
        plt.ylim([0.0, 1.02])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (PR) Curves")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "precision_recall_curves.png"), dpi=300)
        plt.close()
        print("Saved precision_recall_curves.png")

        # 4. Confusion Matrix Heatmap
        stg_res = metrics.get("STG_GBH_Optimal_Threshold", metrics.get("STG_GBH_Default_Threshold", {}))
        if "confusion_matrix" in stg_res:
            cm = np.array(stg_res["confusion_matrix"])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                        xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            ax[0].set_title("Confusion Matrix (Counts)")
            ax[0].set_xlabel("Predicted Label")
            ax[0].set_ylabel("True Label")

            sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax[1],
                        xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
            ax[1].set_title("Normalized Confusion Matrix")
            ax[1].set_xlabel("Predicted Label")
            ax[1].set_ylabel("True Label")

            plt.suptitle("STG-GBH Operational Confusion Matrix", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, "confusion_matrices.png"), dpi=300)
            plt.close()
            print("Saved confusion_matrices.png")

    # 5. Model Comparison Bar Chart
    if len(comp_df) > 0:
        plt.figure(figsize=(12, 6), dpi=300)
        top_models = comp_df.set_index("Model Architecture")[["Accuracy (%)", "Macro-F1 (%)", "Precision (%)", "Recall (%)"]]
        top_models.plot(kind="bar", figsize=(12, 6), colormap="viridis")
        plt.title("Performance Comparison Across Intrusion Detection Architectures")
        plt.ylabel("Score (%)")
        plt.xticks(rotation=45, ha="right")
        plt.ylim([80, 102])
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "model_comparison_bar.png"), dpi=300)
        plt.close()
        print("Saved model_comparison_bar.png")

    # 6. SHAP Feature Importance Plot
    top_feats = metrics.get("Explainability_Top_Features", [])
    if top_feats:
        plt.figure(figsize=(9, 6), dpi=300)
        f_names = [f["feature"] for f in top_feats[:12]][::-1]
        f_vals = [f["mean_abs_shap"] for f in top_feats[:12]][::-1]
        plt.barh(f_names, f_vals, color="#1f77b4", edgecolor="black", alpha=0.8)
        plt.xlabel("Mean |SHAP Value| (Impact on Intrusion Prediction)")
        plt.title("Top Feature Importances via Tree SHAP (STG-GBH Ensemble)")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "feature_importance_shap.png"), dpi=300)
        plt.close()
        print("Saved feature_importance_shap.png")

    # 7. Latency vs Macro-F1 Pareto Chart
    latency_data = []
    for m_name, m in metrics.items():
        if isinstance(m, dict) and "latency_per_1000_obs_ms" in m and "macro_f1" in m:
            latency_data.append({
                "Model": m_name.replace("_", " "),
                "Latency_ms": m["latency_per_1000_obs_ms"],
                "Macro_F1": m["macro_f1"] * 100
            })
    if latency_data:
        lat_df = pd.DataFrame(latency_data)
        plt.figure(figsize=(8, 6), dpi=300)
        plt.scatter(lat_df["Latency_ms"], lat_df["Macro_F1"], color="#d62728", s=100, edgecolors="black", zorder=3)
        for _, row in lat_df.iterrows():
            plt.annotate(row["Model"], (row["Latency_ms"] * 1.05, row["Macro_F1"]), fontsize=9)
        plt.xlabel("Inference Latency per 1,000 Observations (ms)")
        plt.ylabel("Macro-F1 Score (%)")
        plt.title("Computational Efficiency vs Detection Performance Pareto Frontier")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "latency_vs_macro_f1.png"), dpi=300)
        plt.close()
        print("Saved latency_vs_macro_f1.png")

    print("\nAll publication artifacts generated successfully in results/figures/ and results/tables/!")


if __name__ == "__main__":
    generate_all_publication_artifacts("STG_GBH/results")

