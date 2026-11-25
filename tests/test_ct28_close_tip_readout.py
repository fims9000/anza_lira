"""Pure endpoint geometry regression: source CT28 does not need to be opened."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from audit_ct28_close_tip_readout import decode_one
from ccta_3d_an_patches import PatchConfig


def test_three_close_physical_junction_tips_survive_separate_channel_encoding():
    # Geometry-only: real CT28 positives commonly have two tips within 3 mm.
    points=np.array([[5.2,6.1,6.],[7.1,6.1,6.],[6.,8.2,6.]])
    result=decode_one(
        points,PatchConfig(side=24,frame_mode="fixed_lps",endpoint_encoding="separate")
    )
    assert result["degree"]==3
    assert result["min_pair_distance_mm"]<3.
    assert result["old_peaks_valid"] and result["new_peaks_valid"]
    assert result["legacy_best_assignment_max_error_mm"]>0.75
    assert result["new_subvoxel_max_error_mm"]<.05
    assert result["new_all_within_1mm"]


def test_pair_separate_encoding_handles_two_endpoints_without_third_tip():
    points=np.array([[2.,6.,6.],[6.,6.,6.]])
    result=decode_one(
        points,PatchConfig(side=24,frame_mode="fixed_lps",endpoint_encoding="separate")
    )
    assert result["degree"]==2
    assert result["new_peaks_valid"]
    assert result["new_subvoxel_max_error_mm"]<.05
