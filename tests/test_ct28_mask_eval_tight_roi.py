"""Exact full-mask metrics after empty-background crop; no image leakage."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_mask_metrics import active_bbox, cldice, evaluate, dice
from ct28_topology_metrics import gt_component_merge_audit


def original_full_array_cldice(predicted,reference):
    from skimage.morphology import skeletonize
    if not predicted.any() or not reference.any():
        return 1. if not predicted.any() and not reference.any() else 0.
    sp=skeletonize(predicted)
    sr=skeletonize(reference)
    tprec=np.count_nonzero(sp & reference)/np.count_nonzero(sp)
    tsens=np.count_nonzero(sr & predicted)/np.count_nonzero(sr)
    return 2*tprec*tsens/(tprec+tsens) if tprec+tsens else 0.


@pytest.mark.parametrize("edge", [False,True])
def test_exact_crop_equals_uncropped_3d_dice_cldice_and_GT_bridge(edge):
    before=np.zeros((64,63,65),dtype=bool)
    shift=0 if edge else 12
    before[shift:shift+7,18,19]=True
    before[shift+12:shift+20,18,19]=True
    after=before.copy()
    after[shift+7:shift+12,18,19]=True
    gt=after.copy()
    bbox=active_bbox(before,after,gt)
    mask1=before[bbox]; mask2=after[bbox]; truth=gt[bbox]
    assert np.count_nonzero(mask1)==np.count_nonzero(before)
    assert np.count_nonzero(mask2)==np.count_nonzero(after)
    assert np.count_nonzero(truth)==np.count_nonzero(gt)
    report=evaluate(before,after,gt,np.eye(4))
    assert report["dice_before"]==pytest.approx(dice(before,gt))
    assert report["dice_after"]==pytest.approx(dice(after,gt))
    assert report["cldice_before"]==pytest.approx(original_full_array_cldice(before,gt))
    assert report["cldice_after"]==pytest.approx(original_full_array_cldice(after,gt))
    expected=gt_component_merge_audit(before,after,gt)
    assert report["gt_consistent_new_component_pairs"]==expected["gt_consistent_new_component_pairs"]
    assert report["gt_inconsistent_new_component_pairs"]==expected["gt_inconsistent_new_component_pairs"]
    assert report["background_crop_is_exact_for_foreground_metrics"] is True


def test_unchanged_before_after_reuses_same_cldice_and_zero_added():
    x=np.zeros((100,100,100),dtype=bool)
    x[15:20,42,47]=True
    report=evaluate(x,x.copy(),x.copy(),np.eye(4))
    assert report["dice_after"]==report["dice_before"]==1.
    assert report["cldice_after"]==report["cldice_before"]==1.
    assert report["added_voxels"]==0
    assert report["evaluation_tight_crop_bounds_ijk"]==[[13,22],[40,45],[45,50]]


def test_all_empty_mask_does_not_crash_roi():
    blank=np.zeros((10,11,12),dtype=bool)
    r=evaluate(blank,blank.copy(),blank.copy(),np.eye(4))
    assert r["dice_before"]==r["dice_after"]==1.
    assert r["cldice_before"]==r["cldice_after"]==1.
    assert r["reference_components"]==0
    with pytest.raises(ValueError,match="padding"):
        active_bbox(blank,padding=0)
