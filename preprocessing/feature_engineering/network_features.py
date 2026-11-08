"""
Network feature engineering for STG-GBH.
Extracts communication asymmetry, flow rates, TCP window ratios, and protocol characteristics.
"""

import numpy as np
import pandas as pd


def compute_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific network interaction features without lookahead.
    Operates strictly within individual flows.
    """
    feat_df = pd.DataFrame(index=df.index)

    # 1. Asymmetry ratios
    fwd_pkts = None
    bwd_pkts = None
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        fwd_pkts = pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0.0).astype(float)
        bwd_pkts = pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0.0).astype(float)
    elif "IN_PKTS" in df.columns and "OUT_PKTS" in df.columns:
        fwd_pkts = pd.to_numeric(df["IN_PKTS"], errors="coerce").fillna(0.0).astype(float)
        bwd_pkts = pd.to_numeric(df["OUT_PKTS"], errors="coerce").fillna(0.0).astype(float)
    elif "tcp.srcport" in df.columns and "tcp.dstport" in df.columns:
        fwd_pkts = pd.to_numeric(df.get("tcp.seq", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0).clip(lower=1.0)
        bwd_pkts = pd.to_numeric(df.get("tcp.ack", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0).clip(lower=1.0)

    if fwd_pkts is not None and bwd_pkts is not None:
        total_pkts = fwd_pkts + bwd_pkts + 1e-6
        feat_df["net_packet_asymmetry_ratio"] = (fwd_pkts - bwd_pkts) / total_pkts
        feat_df["net_fwd_to_bwd_ratio"] = fwd_pkts / (bwd_pkts + 1.0)
        feat_df["net_log_total_pkts"] = np.log1p(np.maximum(0.0, fwd_pkts + bwd_pkts))

    # 2. Byte asymmetry
    fwd_bytes = None
    bwd_bytes = None
    if "Fwd Packets Length Total" in df.columns and "Bwd Packets Length Total" in df.columns:
        fwd_bytes = pd.to_numeric(df["Fwd Packets Length Total"], errors="coerce").fillna(0.0).astype(float)
        bwd_bytes = pd.to_numeric(df["Bwd Packets Length Total"], errors="coerce").fillna(0.0).astype(float)
    elif "IN_BYTES" in df.columns and "OUT_BYTES" in df.columns:
        fwd_bytes = pd.to_numeric(df["IN_BYTES"], errors="coerce").fillna(0.0).astype(float)
        bwd_bytes = pd.to_numeric(df["OUT_BYTES"], errors="coerce").fillna(0.0).astype(float)
    elif "http.content_length" in df.columns and "tcp.len" in df.columns:
        fwd_bytes = pd.to_numeric(df["http.content_length"], errors="coerce").fillna(0.0).astype(float)
        bwd_bytes = pd.to_numeric(df["tcp.len"], errors="coerce").fillna(0.0).astype(float)

    if fwd_bytes is not None and bwd_bytes is not None:
        tot_bytes = fwd_bytes + bwd_bytes + 1e-6
        feat_df["net_byte_asymmetry_ratio"] = (fwd_bytes - bwd_bytes) / tot_bytes
        feat_df["net_fwd_to_bwd_byte_ratio"] = fwd_bytes / (bwd_bytes + 1.0)
        feat_df["net_log_total_bytes"] = np.log1p(np.maximum(0.0, fwd_bytes + bwd_bytes))
        if fwd_pkts is not None:
            feat_df["net_bytes_per_fwd_pkt"] = fwd_bytes / (fwd_pkts + 1.0)
            feat_df["net_bytes_per_bwd_pkt"] = bwd_bytes / (bwd_pkts + 1.0)

    # 3. Packet size variance and spread
    if "Fwd Packet Length Max" in df.columns and "Fwd Packet Length Min" in df.columns:
        feat_df["net_fwd_pkt_len_spread"] = (
            pd.to_numeric(df["Fwd Packet Length Max"], errors="coerce").fillna(0.0) -
            pd.to_numeric(df["Fwd Packet Length Min"], errors="coerce").fillna(0.0)
        )
    if "Bwd Packet Length Max" in df.columns and "Bwd Packet Length Min" in df.columns:
        feat_df["net_bwd_pkt_len_spread"] = (
            pd.to_numeric(df["Bwd Packet Length Max"], errors="coerce").fillna(0.0) -
            pd.to_numeric(df["Bwd Packet Length Min"], errors="coerce").fillna(0.0)
        )

    # 4. TCP Window and Header ratios
    if "Init Fwd Win Bytes" in df.columns and "Init Bwd Win Bytes" in df.columns:
        fw_win = pd.to_numeric(df["Init Fwd Win Bytes"], errors="coerce").fillna(0.0).clip(lower=0.0)
        bw_win = pd.to_numeric(df["Init Bwd Win Bytes"], errors="coerce").fillna(0.0).clip(lower=0.0)
        feat_df["net_win_byte_ratio"] = (fw_win + 1.0) / (bw_win + 1.0)
        feat_df["net_log_fwd_win"] = np.log1p(fw_win)
        feat_df["net_log_bwd_win"] = np.log1p(bw_win)

    # 5. Flag interaction (SYN without ACK, RST frequency)
    if "SYN Flag Count" in df.columns and "ACK Flag Count" in df.columns:
        syn = pd.to_numeric(df["SYN Flag Count"], errors="coerce").fillna(0.0)
        ack = pd.to_numeric(df["ACK Flag Count"], errors="coerce").fillna(0.0)
        feat_df["net_syn_ack_discrepancy"] = syn - ack
    elif "tcp.connection.syn" in df.columns and "tcp.connection.synack" in df.columns:
        syn = pd.to_numeric(df["tcp.connection.syn"], errors="coerce").fillna(0.0)
        synack = pd.to_numeric(df["tcp.connection.synack"], errors="coerce").fillna(0.0)
        feat_df["net_syn_ack_discrepancy"] = syn - synack

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat_df
