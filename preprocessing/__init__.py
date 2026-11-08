"""Preprocessing module for STG-GBH."""
from .label_mapping import map_raw_label, CANONICAL_FAMILIES, MITRE_STAGES
from .common_preprocessing import LeakageFreePreprocessor, chronological_split
from .duplicate_detection import audit_split_duplicates, purge_cross_split_leakage
from .feature_validation import FeatureValidator, clean_infinities_and_nans

