"""Frozen development-validation corruption uses same geometry as train, never test."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_train_gap_simulator import (
    synthesize_train_gaps,synthesize_validation_gaps,
    VAL_CT28_IDS
)


def tube():
    mask=np.zeros((90,24,24),dtype=bool)
    mask[8:80,12,12]=True
    return mask


def test_same_frozen_corruption_algorithm_but_separate_validation_receipt():
    original=tube()
    train,train_receipt=synthesize_train_gaps(
        original,np.eye(4),patient_id=953,gaps=3,
        radius_mm=1.1,seed=42,
    )
    for validation in sorted(VAL_CT28_IDS):
        val,receipt=synthesize_validation_gaps(
            original,np.eye(4),patient_id=validation,
            gaps=3,radius_mm=1.1,seed=42,
        )
        assert np.array_equal(val,train)
        assert receipt["protocol"]=="ct28_validation_controlled_mask_gap_v1"
        assert receipt["patient_id"]==validation
        assert receipt["damaged_components"]==4
        assert receipt["centers"]==train_receipt["centers"]


def test_validation_generator_cannot_use_train_or_test():
    for patient in (953,954,955,973,984):
        with pytest.raises(ValueError,match="five CT28 validation"):
            synthesize_validation_gaps(
                tube(),np.eye(4),patient_id=patient,
                gaps=3,radius_mm=1.1,
            )
