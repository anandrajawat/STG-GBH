"""Models module for STG-GBH."""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import xgboost
import lightgbm
import catboost
import torch
torch.set_num_threads(1)

from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .stacking_ensemble import OOFStackingEnsemble
from .graphsage import HeteroGraphSAGE, HomogeneousGraphSAGE
from .temporal_gru import TemporalGRU
from .temporal_attention import TemporalAttention
from .fusion_model import STGGBHFusionModel
from .multitask_heads import MultiTaskPredictionHeads, UncertaintyWeightedLoss
