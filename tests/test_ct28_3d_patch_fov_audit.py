"""Actual CT28 3D endpoint FOV audit on committed geometry, never raw CT."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_patches import PatchConfig
from ct28_3d_patch_fov_audit import endpoint_margin_mm,run


def test_physical_cube_containment_is_signed_and_rotation_safe():
    cfg=PatchConfig(side=24,voxel_mm=1.)
    pair_ok=np.array([[0.,0.,0.],[20.,0.,0.]])
    pair_far=np.array([[0.,0.,0.],[30.,0.,0.]])
    assert endpoint_margin_mm(pair_ok,cfg)==pytest.approx(1.5)
    assert endpoint_margin_mm(pair_far,cfg)==pytest.approx(-3.5)
    assert endpoint_margin_mm(pair_ok[::-1],cfg)==pytest.approx(1.5)


def test_all_15092_real_CT28_endpoints_fit_original_24mm_patch():
    report,frames=run(perturbations=1,seed=20260920)
    assert report["candidate_scenes"]==1404
    assert report["source_candidate_rows"]==15092
    assert len(frames)==15092*2
    assert report["held_out_test_accessed"] is False
    assert set(frames.split)=={"train","val"}
    for kind,split,n in (
        ("PAIR","train",6336),
        ("PAIR","val",2298),
        ("JUNCTION","train",4685),
        ("JUNCTION","val",1773),
    ):
        group=report["groups"][f"{split}/{kind}/clean"]
        assert group["samples"]==n
        assert group["outside_patch"]==0
        assert group["min_margin_mm"]>0
    assert sum(x["samples"] for x in report["groups"].values())==2*15092
    assert all(x["samples"]>0 for x in report["groups"].values())
