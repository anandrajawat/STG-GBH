"""
Automated unit tests for models, ensembles, and fusion inference in STG-GBH.
"""

import torch
import numpy as np
import pytest

from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from models.stacking_ensemble import OOFStackingEnsemble
from models.temporal_gru import TemporalGRU
from models.temporal_attention import TemporalAttention
from models.fusion_model import STGGBHFusionModel


def test_boosting_models_inference():
    X = np.random.randn(50, 10)
    y = np.random.choice([0, 1], size=50)

    # XGBoost
    xgb_m = XGBoostModel(params={"n_estimators": 5}, task="binary")
    xgb_m.fit(X, y)
    p_xgb = xgb_m.predict_proba(X)
    assert p_xgb.shape == (50, 2)
    assert np.allclose(p_xgb.sum(axis=1), 1.0, atol=1e-5)

    # CatBoost
    cat_m = CatBoostModel(params={"iterations": 5}, task="binary")
    cat_m.fit(X, y)
    p_cat = cat_m.predict_proba(X)
    assert p_cat.shape == (50, 2)

    # LightGBM
    lgb_m = LightGBMModel(params={"n_estimators": 5}, task="binary")
    lgb_m.fit(X, y)
    p_lgb = lgb_m.predict_proba(X)
    assert p_lgb.shape == (50, 2)


def test_oof_stacking_ensemble():
    X = np.random.randn(60, 8)
    y = np.random.choice([0, 1], size=60)
    stack = OOFStackingEnsemble(task="binary", n_splits=3)
    stack.fit(X, y)

    p_stack = stack.predict_proba(X)
    assert p_stack.shape == (60, 2)
    assert np.allclose(p_stack.sum(axis=1), 1.0, atol=1e-5)

    voting = stack.predict_soft_voting(X)
    assert voting.shape == (60, 2)


def test_temporal_gru_and_attention():
    seq = torch.randn(8, 12, 64) # batch=8, seq_len=12, dim=64
    gru = TemporalGRU(embed_dim=64, hidden_dim=64, num_layers=1)
    out, h_n = gru(seq)
    assert out.shape == (8, 12, 64)

    attn = TemporalAttention(hidden_dim=64, mode="additive")
    context, weights = attn(out)
    assert context.shape == (8, 64)
    assert weights.shape == (8, 12)
    assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)


def test_fusion_model_forward():
    h_graph = torch.randn(10, 64)
    p_boost = torch.rand(10, 4) # 4 probability features

    fusion = STGGBHFusionModel(
        graph_embed_dim=64,
        num_boosting_prob_cols=4,
        hidden_dim=64,
        num_classes=2,
        enable_multitask=True,
        num_categories=5,
        num_mitre_stages=4
    )

    logits, mt_outs, z = fusion(h_graph, p_boost)
    assert logits.shape == (10, 2)
    assert mt_outs["category_logits"].shape == (10, 5)
    assert mt_outs["mitre_logits"].shape == (10, 4)
    assert z.shape == (10, 64)

