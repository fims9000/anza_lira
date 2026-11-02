from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_mask_metrics import dice, cldice, evaluate


def test_rasterization_improves_dice_and_cldice_on_synthetic_gap():
    ref = np.zeros((18, 18, 18), dtype=bool)
    ref[2:14, 8, 8] = True
    before = ref.copy()
    before[7:10, 8, 8] = False
    after = ref.copy()
    score = evaluate(before, after, ref, np.diag([1, 1, 2, 1]))
    assert score["dice_after"] == 1.0
    assert score["cldice_after"] == 1.0
    assert score["dice_before"] < 1.0
    assert score["cldice_before"] < 1.0
    assert score["original_components"] == 2
    assert score["final_components"] == 1
    assert score["added_voxels"] == 3
    assert score["added_volume_mm3"] == 6.0
    assert score["reference_unsupported_added_voxels"] == 0


def test_outside_groundtruth_additions_are_recorded():
    ref = np.zeros((8, 8, 8), dtype=bool)
    ref[2:4, 3, 3] = True
    before = ref.copy()
    after = ref.copy()
    after[4, 3, 3] = True
    score = evaluate(before, after, ref, np.eye(4))
    assert score["reference_unsupported_added_voxels"] == 1
    assert score["added_voxel_precision_vs_reference"] == 0
    assert score["dice_after"] < score["dice_before"]


def test_empty_masks_and_destructive_change():
    mask = np.zeros((8, 8, 8), dtype=bool)
    assert dice(mask, mask) == 1
    assert cldice(mask, mask) == 1
    before = mask.copy()
    before[2, 3, 3] = True
    with pytest.raises(ValueError, match="removed"):
        evaluate(before, mask, before, np.eye(4))
