"""
Graph feature engineering for STG-GBH.
Calculates topological degree and interaction frequencies from past graph interactions.
"""

import numpy as np
import pandas as pd


def compute_topological_graph_features(
    df: pd.DataFrame,
    src_col: str = "ip.src_host",
    dst_col: str = "ip.dst_host",
    proto_col: str = "Protocol"
) -> pd.DataFrame:
    """
    Computes node interaction degree and connectivity features strictly within each session.
    If IP columns are unavailable (e.g. CSE-CIC-IDS2018 stripped), computes protocol-window-segment
    endpoint signature degrees.
    """
    feat_df = pd.DataFrame(index=df.index)
    sess = df["session_id"].astype(str) if "session_id" in df.columns else pd.Series("s0", index=df.index)

    if src_col in df.columns and dst_col in df.columns:
        src_series = sess + "_" + df[src_col].astype(str)
        dst_series = sess + "_" + df[dst_col].astype(str)

        feat_df["graph_src_historical_out_degree"] = src_series.groupby(src_series).cumcount()
        feat_df["graph_dst_historical_in_degree"] = dst_series.groupby(dst_series).cumcount()

        pair_series = src_series + "->" + dst_series
        feat_df["graph_edge_historical_freq"] = pair_series.groupby(pair_series).cumcount()

        feat_df["graph_degree_ratio"] = (feat_df["graph_src_historical_out_degree"] + 1.0) / (
            feat_df["graph_dst_historical_in_degree"] + 1.0
        )
    else:
        p_col = proto_col if proto_col in df.columns else ("Protocol Type" if "Protocol Type" in df.columns else ("PROTOCOL" if "PROTOCOL" in df.columns else None))
        if p_col is not None:
            p_series = sess + "_" + df[p_col].astype(str)
            feat_df["graph_proto_historical_freq"] = p_series.groupby(p_series).cumcount()
            # Service-stack interaction signature (Protocol + Fwd Seg Size Min + Init Fwd Win Bytes bucket)
            if "Fwd Seg Size Min" in df.columns and "Init Fwd Win Bytes" in df.columns:
                seg_s = df["Fwd Seg Size Min"].astype(str)
                win_s = (pd.to_numeric(df["Init Fwd Win Bytes"], errors="coerce").fillna(-1).astype(int) // 256).astype(str)
                endpoint_sig = p_series + "_" + seg_s + "_" + win_s
                # Causal rolling recurrence of endpoint stack signature in last 25 flows
                sig_same = (endpoint_sig == endpoint_sig.groupby(sess).shift(1)).astype(float)
                feat_df["graph_endpoint_stack_persistence"] = (
                    sig_same.groupby(sess).rolling(25, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)
                )
        else:
            feat_df["graph_proto_historical_freq"] = 0.0

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat_df
