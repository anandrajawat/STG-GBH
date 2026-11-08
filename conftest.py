"""Pytest root configuration ensuring OpenMP safety on macOS arm64."""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import xgboost
import catboost
import lightgbm
import torch
torch.set_num_threads(1)
