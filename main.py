"""
Master Pipeline Orchestrator for STG-GBH Framework.
Spatio-Temporal Graph Neural Network with Optimized XGBoost, CatBoost,
and LightGBM Ensemble for Cross-Domain Cybersecurity Intrusion Detection.
"""

import os
import sys
import argparse
import time

# Ensure OpenMP stability on macOS
os.environ["OMP_NUM_THREADS"] = "1"
import xgboost
import catboost
import lightgbm
import torch
torch.set_num_threads(1)

# Ensure local packages are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.build_audit_report import generate_audit_report
from scripts.run_leakage_audit import run_full_leakage_audit, generate_leakage_audit_markdown
from scripts.run_hyperparameter_optimization import run_all_optimizations
from experiments.in_domain import InDomainExperiment
from experiments.cross_dataset import CrossDatasetExperiment
from experiments.cross_domain import CrossDomainExperiment
from experiments.ablation import AblationExperiment
from scripts.run_statistical_validation import run_statistical_significance_suite
from scripts.generate_publication_figures import generate_all_publication_artifacts


def main():
    parser = argparse.ArgumentParser(description="STG-GBH Cybersecurity Research Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "audit", "leakage", "optimize", "indomain", "cross_dataset", "cross_domain", "ablation", "figures", "reports"],
        help="Pipeline execution mode"
    )
    parser.add_argument("--samples_per_day", type=int, default=4000, help="Sample size per capture session")
    parser.add_argument("--optuna_trials", type=int, default=15, help="Number of Optuna TPE trials per model")
    args = parser.parse_args()

    t_start = time.time()
    print("==========================================================================")
    print("  STG-GBH: SPATIO-TEMPORAL GRAPH AND BOOSTING HYBRID PIPELINE            ")
    print("  Decoupled Spatio-Temporal Heterogeneous Graph Learning with            ")
    print("  Optimized Gradient Boosting Ensembles for Cross-Domain Intrusion Detection")
    print("==========================================================================")

    if args.mode in ["full", "audit"]:
        print("\n>>> PHASE 1: DATASET AUDIT & PROVENANCE VERIFICATION <<<")
        generate_audit_report(current_dir)
        print("Dataset audit report verified in reports/dataset_audit.md.")

    if args.mode in ["full", "leakage"]:
        print("\n>>> PHASE 2: SCIENTIFIC DATA LEAKAGE AUDIT <<<")
        leak_res = run_full_leakage_audit(os.path.join(current_dir, "datasets/CSE_CIC_IDS2018"))
        generate_leakage_audit_markdown(leak_res, os.path.join(current_dir, "reports/leakage_audit.md"))
        print("Data leakage audit completed and verified in reports/leakage_audit.md.")

    if args.mode in ["full", "optimize"]:
        print("\n>>> PHASE 3: OPTUNA TPE HYPERPARAMETER OPTIMIZATION <<<")
        run_all_optimizations(
            data_dir=os.path.join(current_dir, "datasets/CSE_CIC_IDS2018"),
            n_trials=args.optuna_trials,
            output_dir=os.path.join(current_dir, "results/optimization")
        )

    if args.mode in ["full", "indomain"]:
        print("\n>>> PHASE 4: IN-DOMAIN BENCHMARKING & COMPLETE STG-GBH TRAINING <<<")
        indomain_exp = InDomainExperiment(
            data_dir=os.path.join(current_dir, "datasets/CSE_CIC_IDS2018"),
            output_dir=os.path.join(current_dir, "results"),
            samples_per_day=args.samples_per_day
        )
        indomain_exp.run()
        run_statistical_significance_suite(os.path.join(current_dir, "results"))

    if args.mode in ["full", "cross_dataset"]:
        print("\n>>> PHASE 5: CROSS-DATASET EVALUATION (CSE-2018 -> CIC-2017) <<<")
        cross_exp = CrossDatasetExperiment(
            c18_dir=os.path.join(current_dir, "datasets/CSE_CIC_IDS2018"),
            c17_dir=os.path.join(current_dir, "datasets/CIC_IDS2017"),
            output_dir=os.path.join(current_dir, "results"),
            samples_per_file=args.samples_per_day
        )
        cross_exp.run()

    if args.mode in ["full", "cross_domain"]:
        print("\n>>> PHASE 6: CROSS-DOMAIN IOT EVALUATION <<<")
        iot_exp = CrossDomainExperiment(
            edge_path=os.path.join(current_dir, "datasets/Edge_IIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"),
            ciciot_dir=os.path.join(current_dir, "datasets/CICIoT2023"),
            ton_path=os.path.join(current_dir, "datasets/ToN_IoT/NF-ToN-IoT-V2.parquet"),
            output_dir=os.path.join(current_dir, "results")
        )
        iot_exp.run()

    if args.mode in ["full", "ablation"]:
        print("\n>>> PHASE 7: 15-FACTOR ABLATION STUDIES <<<")
        ablation_exp = AblationExperiment(
            data_dir=os.path.join(current_dir, "datasets/CSE_CIC_IDS2018"),
            output_dir=os.path.join(current_dir, "results"),
            samples_per_day=args.samples_per_day
        )
        ablation_exp.run()

    if args.mode in ["full", "figures"]:
        print("\n>>> PHASE 8: GENERATING PUBLICATION FIGURES & TABLES <<<")
        generate_all_publication_artifacts(os.path.join(current_dir, "results"))

    total_elapsed = time.time() - t_start
    print(f"\nPipeline finished successfully in {total_elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
