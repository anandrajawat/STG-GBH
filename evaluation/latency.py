"""
Inference latency and throughput benchmarking for STG-GBH on Apple Silicon M2.
"""

import time
import numpy as np
from typing import Dict, Any, Callable


def benchmark_inference_latency(
    predict_fn: Callable[[np.ndarray], Any],
    sample_batch: np.ndarray,
    n_warmup: int = 5,
    n_repeats: int = 20
) -> Dict[str, float]:
    """
    Measures complete end-to-end inference latency on test observations.
    Calculates mean, median, p95, latency per 1,000 observations, and throughput.
    """
    batch_size = len(sample_batch)

    # Warmup
    for _ in range(n_warmup):
        _ = predict_fn(sample_batch)

    latencies = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = predict_fn(sample_batch)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    throughput = float(batch_size / mean_lat) if mean_lat > 0 else 0.0
    lat_per_1000 = float((mean_lat / batch_size) * 1000.0)

    return {
        "batch_size": batch_size,
        "mean_latency_ms": round(mean_lat * 1000.0, 3),
        "median_latency_ms": round(median_lat * 1000.0, 3),
        "p95_latency_ms": round(p95_lat * 1000.0, 3),
        "p99_latency_ms": round(p99_lat * 1000.0, 3),
        "latency_per_1000_obs_ms": round(lat_per_1000 * 1000.0, 3),
        "throughput_obs_per_sec": round(throughput, 1)
    }

