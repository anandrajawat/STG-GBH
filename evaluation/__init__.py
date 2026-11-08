"""Evaluation module for STG-GBH."""
from .metrics import compute_comprehensive_metrics
from .calibration import compute_calibration_metrics
from .latency import benchmark_inference_latency
from .memory import get_process_memory_mb, compute_model_footprint
from .explainability import compute_tree_shap_summary, compare_feature_importances
from .error_analysis import perform_error_analysis

