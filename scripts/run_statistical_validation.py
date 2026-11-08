"""
Runs statistical significance testing, McNemar test, and 95% CIs on STG-GBH predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from experiments.statistical_validation import compute_mcnemar_test, compute_confidence_intervals, compute_cohens_d


def run_statistical_significance_suite(results_dir: str = "results"):
    pred_path = os.path.join(results_dir, "predictions/in_domain_test_predictions.parquet")
    if not os.path.exists(pred_path):
        print(f"Predictions file not found: {pred_path}")
        return

    df = pd.read_parquet(pred_path)
    y_true = df["y_true"].values
    pred_stg = df["pred_stg_gbh_optimal"].values

    base_preds = {
        "Random_Forest": (df["p_rf"].values >= 0.5).astype(int),
        "XGBoost": (df["p_xgb"].values >= 0.5).astype(int),
        "CatBoost": (df["p_cat"].values >= 0.5).astype(int),
        "LightGBM": (df["p_lgb"].values >= 0.5).astype(int),
        "OOF_Stacking": (df["p_stack"].values >= 0.5).astype(int)
    }

    report = {"mcnemar_tests": {}, "bootstrap_confidence_intervals": {}}

    for name, p_base in base_preds.items():
        mcnemar = compute_mcnemar_test(y_true, pred_stg, p_base)
        report["mcnemar_tests"][f"STG_GBH_vs_{name}"] = mcnemar
        print(f"McNemar test vs {name}: Chi2={mcnemar['chi2_statistic']}, p={mcnemar['p_value']} (Sig: {mcnemar['significant_at_05']})")

    # Bootstrap 95% CI on STG-GBH Macro-F1 and Accuracy
    from sklearn.metrics import f1_score, accuracy_score
    n_bootstraps = 500
    rng = np.random.RandomState(42)
    boot_f1s = []
    boot_accs = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        boot_f1s.append(float(f1_score(y_true[idx], pred_stg[idx], average="macro", zero_division=0)))
        boot_accs.append(float(accuracy_score(y_true[idx], pred_stg[idx])))

    report["bootstrap_confidence_intervals"]["STG_GBH_Macro_F1_95CI"] = compute_confidence_intervals(boot_f1s)
    report["bootstrap_confidence_intervals"]["STG_GBH_Accuracy_95CI"] = compute_confidence_intervals(boot_accs)

    out_file = os.path.join(results_dir, "metrics/statistical_validation.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    # Save summary table
    rows = []
    for comp_name, m in report["mcnemar_tests"].items():
        rows.append({
            "Hypothesis Comparison": comp_name.replace("_", " "),
            "Chi2 Statistic": m["chi2_statistic"],
            "p-Value": m["p_value"],
            "Statistically Significant (p < 0.05)": m["significant_at_05"],
            "STG-GBH Superior Cases": m["b_model_a_better"],
            "Baseline Superior Cases": m["c_model_b_better"]
        })
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "tables/statistical_significance_table.csv"), index=False)
    print("Statistical significance analysis completed successfully!")


if __name__ == "__main__":
    run_statistical_significance_suite("STG_GBH/results")

