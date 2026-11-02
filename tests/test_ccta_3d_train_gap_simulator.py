"""Only controlled synthetic 3D tubes; no real medical or test data."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_train_gap_simulator import synthesize_train_gaps, TRAIN_CT28_IDS


def tube():
    volume = np.zeros((90, 24, 24), dtype=bool)
    volume[8:80, 12, 12] = True
    return volume


def test_reproducible_multi_gap_3d_train_corruption():
    original = tube()
    damaged, report = synthesize_train_gaps(
        original, np.eye(4), patient_id=953, gaps=3, radius_mm=1.1,
        seed=20261102,
    )
    repeated, second = synthesize_train_gaps(
        original, np.eye(4), patient_id=953, gaps=3, radius_mm=1.1,
        seed=20261102,
    )
    assert np.array_equal(damaged, repeated)
    assert report == second
    assert report["original_components"] == 1
    assert report["damaged_components"] == 4
    assert int(damaged.sum()) == int(original.sum()) - report["removed_voxels"]
    assert np.all(damaged <= original)
    assert np.array_equal(original, tube())


def test_held_out_and_validation_not_permitted():
    original = tube()
    for patient in (954, 957, 961, 973):
        with pytest.raises(ValueError, match="train"):
            synthesize_train_gaps(original, np.eye(4), patient_id=patient,
                                  gaps=3, radius_mm=1.1)


def test_invalid_gap_plan_fails_without_partial_success():
    mask = np.zeros((18, 18, 18), dtype=bool)
    mask[5:12, 8, 8] = True
    with pytest.raises(ValueError):
        synthesize_train_gaps(mask, np.eye(4), patient_id=953, gaps=3,
                              radius_mm=3)
