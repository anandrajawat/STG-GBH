"""
Automated unit tests for data integrity, label mapping, and schema consistency in STG-GBH.
"""

import pytest
import numpy as np
import pandas as pd

from preprocessing.label_mapping import map_raw_label, CANONICAL_FAMILIES, MITRE_STAGES
from preprocessing.feature_validation import FeatureValidator, clean_infinities_and_nans
from preprocessing.duplicate_detection import compute_row_hashes, audit_split_duplicates


def test_label_mapping_canonical():
    is_att, fam, stage, _ = map_raw_label("DoS attacks-Hulk", "CSE_CIC_IDS2018")
    assert is_att == 1
    assert fam == "DOS"
    assert stage == "IMPACT"

    is_att_b, fam_b, stage_b, _ = map_raw_label("Benign", "CSE_CIC_IDS2018")
    assert is_att_b == 0
    assert fam_b == "BENIGN"
    assert stage_b == "BENIGN"


def test_feature_validator_forbids_target():
    # Target in feature names should trigger exception
    bad_features = ["Flow_Duration", "Attack_Label", "Bytes"]
    with pytest.raises(ValueError):
        _ = FeatureValidator(bad_features)


def test_clean_infinities_and_nans():
    df = pd.DataFrame({
        "feat1": [1.0, np.inf, -np.inf, np.nan, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    clean_df, medians = clean_infinities_and_nans(df, ["feat1", "feat2"])
    assert not np.isinf(clean_df.values).any()
    assert not np.isnan(clean_df.values).any()
    assert medians["feat1"] == 3.0


def test_duplicate_hashing_and_isolation():
    df_tr = pd.DataFrame({"f1": [1.0, 2.0], "f2": [10.0, 20.0]})
    df_val = pd.DataFrame({"f1": [1.0, 3.0], "f2": [10.0, 30.0]})
    df_te = pd.DataFrame({"f1": [4.0, 5.0], "f2": [40.0, 50.0]})

    res = audit_split_duplicates(df_tr, df_val, df_te, ["f1", "f2"])
    assert res["train_val_leakage_hashes"] == 1
    assert res["train_test_leakage_hashes"] == 0
    assert not res["is_leakage_free"]

