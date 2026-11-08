"""
Behavioral and temporal feature engineering for STG-GBH.
Constructs strictly backward-looking causal rolling statistics within each session
to eliminate future lookahead while capturing burst and stealth attack dynamics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


CAUSAL_BEHAVIORAL_CANDIDATES = [
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Seg Size Min",
    "PSH Flag Count",
    "ACK Flag Count",
    "SYN Flag Count",
    "Protocol",
    # Alternative dataset column names (IoT / NetFlow)
    "Rate",
    "IN_PKTS",
    "OUT_PKTS",
    "IN_BYTES",
    "OUT_BYTES",
    "FLOW_DURATION_MILLISECONDS",
    "tcp.len",
]


def compute_causal_temporal_features(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    window_sizes: List[int] = [10, 30, 50]
) -> pd.DataFrame:
    """
    Computes causal temporal features using backward-shifted rolling windows grouped by session.
    Strictly forbids looking at the current row or future rows for the window aggregation.
    """
    feat_df = pd.DataFrame(index=df.index)
    grp = df["session_id"] if "session_id" in df.columns else pd.Series(0, index=df.index)

    # 1. Inter-arrival time (delta t since previous event)
    if time_col and time_col in df.columns:
        times = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)
        dt = times.groupby(grp).diff().clip(lower=0.0).fillna(0.0)
        feat_df["causal_inter_arrival_time"] = dt
        feat_df["causal_log_inter_arrival_time"] = np.log1p(dt)
    else:
        feat_df["causal_inter_arrival_time"] = 1.0
        feat_df["causal_log_inter_arrival_time"] = 0.693

    # 2. Primary rate column for backward compatibility with SHAP feature naming
    rate_col = None
    for cand in ["Flow Packets/s", "Flow Bytes/s", "Rate", "IN_PKTS", "tcp.len"]:
        if cand in df.columns:
            rate_col = cand
            break

    if rate_col is not None:
        values = pd.to_numeric(df[rate_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        shifted = values.groupby(grp).shift(1).fillna(0.0)
        for w in window_sizes:
            rolling_mean = shifted.groupby(grp).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_std = shifted.groupby(grp).rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            feat_df[f"causal_past_rate_mean_{w}"] = rolling_mean
            feat_df[f"causal_past_rate_std_{w}"] = rolling_std
            feat_df[f"causal_rate_deviation_{w}"] = values - rolling_mean

    # 3. Multi-variate causal rolling statistics across behavioral flow indicators
    active_cols = [c for c in CAUSAL_BEHAVIORAL_CANDIDATES if c in df.columns and c != rate_col]
    for col in active_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        shifted = s.groupby(grp).shift(1).fillna(0.0)
        for w in window_sizes:
            r_mean = shifted.groupby(grp).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            r_std = shifted.groupby(grp).rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
            feat_df[f"causal_{col}_mean_{w}"] = r_mean
            feat_df[f"causal_{col}_std_{w}"] = r_std

    # 4. Entity destination diversity if entity column exists
    if entity_col and entity_col in df.columns and "ip.dst_host" in df.columns:
        dst_s = df["ip.dst_host"].astype(str)
        shifted_dst = dst_s.groupby(grp).shift(1).fillna("none")
        for w in [20, 50]:
            diffs = (shifted_dst != shifted_dst.groupby(grp).shift(1)).astype(float)
            feat_df[f"causal_dst_change_freq_{w}"] = (
                diffs.groupby(grp).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            )

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat_df
