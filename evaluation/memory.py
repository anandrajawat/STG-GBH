"""
Memory profiling and model footprint metrics for STG-GBH.
"""

import os
import psutil
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))


def compute_model_footprint(
    model: Optional[nn.Module] = None,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes parameter count, trainable parameter count, and storage size on disk.
    """
    footprint = {}
    if model is not None and isinstance(model, nn.Module):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        footprint["total_parameters"] = int(total_params)
        footprint["trainable_parameters"] = int(trainable_params)
        footprint["in_memory_param_size_mb"] = round(float(param_size_mb), 3)

    if checkpoint_path and os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        footprint["checkpoint_size_mb"] = round(float(size_mb), 3)

    return footprint

