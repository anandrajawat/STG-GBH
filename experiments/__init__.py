"""Experiments module for STG-GBH."""
from .in_domain import InDomainExperiment
from .cross_dataset import CrossDatasetExperiment
from .cross_domain import CrossDomainExperiment
from .ablation import AblationExperiment
from .statistical_validation import StatisticalValidationRunner, compute_mcnemar_test

