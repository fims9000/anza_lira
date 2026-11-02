"""Medical-mask posthoc component topology invariants on synthetic vessels.

GT is never passed to any inference, only metrics. A good Dice score alone
cannot certify that a repair did not bridge unrelated arterial trees.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_topology_metrics import gt_component_merge_audit
from ct28_mask_metrics import evaluate


def pieces():
    before=np.zeros((16,10,10),dtype=bool)
    before[2:5,5,5]=True
    before[10:13,5,5]=True
    after=before.copy()
    after[5:10,5,5]=True
    return before,after


def test_correct_same_GT_component_merge_and_dice():
    before,after=pieces()
    gt=after.copy()
    outcome=gt_component_merge_audit(before,after,gt)
    assert outcome["new_original_component_pairs_connected"]==1
    assert outcome["gt_consistent_new_component_pairs"]==1
    assert outcome["gt_inconsistent_new_component_pairs"]==0
    assert outcome["gt_unassigned_new_component_pairs"]==0
    assert outcome["reference_component_count"]==1
    assert outcome["result_component_count"]==1
    metrics=evaluate(before,after,gt,np.eye(4))
    assert metrics["dice_after"]>metrics["dice_before"]
    assert metrics["gt_consistent_new_component_pairs"]==1


def test_bridging_distinct_GT_trees_is_an_explicit_false_bridge():
    before,after=pieces()
    gt=before.copy()
    report=gt_component_merge_audit(before,after,gt)
    assert report["reference_component_count"]==2
    assert report["gt_consistent_new_component_pairs"]==0
    assert report["gt_inconsistent_new_component_pairs"]==1
    flagged=report["per_merge_examples"][0]
    assert flagged["gt_inconsistent_pairs"]==1
    assert len(flagged["before_component_representative_ijk"])==2
    assert report["gt_unassigned_new_component_pairs"]==0
    metrics=evaluate(before,after,gt,np.eye(4))
    assert metrics["reference_unsupported_added_voxels"]>0
    assert metrics["gt_inconsistent_new_component_pairs"]==1


def test_merged_segment_without_any_gt_overlap_is_unknown_not_safe():
    before,after=pieces()
    gt=np.zeros_like(before)
    gt[2:5,5,5]=True
    report=gt_component_merge_audit(before,after,gt)
    assert report["gt_consistent_new_component_pairs"]==0
    assert report["gt_inconsistent_new_component_pairs"]==0
    assert report["gt_unassigned_new_component_pairs"]==1
    assert report["original_components_without_GT_overlap"]==1


def test_unchanged_mask_does_not_invent_pairs_and_never_removes_voxels():
    before,_=pieces()
    audit=gt_component_merge_audit(before,before.copy(),before.copy())
    assert audit["new_original_component_pairs_connected"]==0
    assert audit["gt_consistent_new_component_pairs"]==0
    assert audit["gt_inconsistent_new_component_pairs"]==0
    assert audit["gt_unassigned_new_component_pairs"]==0
    removed=before.copy()
    removed[2,5,5]=False
    with pytest.raises(ValueError,match="preserve"):
        gt_component_merge_audit(before,removed,before)


def test_gt_component_pairs_scale_as_n_choose_2_not_just_number_of_merges():
    mask=np.zeros((20,7,7),dtype=bool)
    mask[2:4,3,3]=True
    mask[8:10,3,3]=True
    mask[16:18,3,3]=True
    after=mask.copy()
    after[4:17,3,3]=True
    gt=np.zeros_like(mask)
    gt[2:10,3,3]=True
    gt[16:18,3,3]=True
    m=gt_component_merge_audit(mask,after,gt)
    assert m["new_original_component_pairs_connected"]==3
    assert m["gt_consistent_new_component_pairs"]==1
    assert m["gt_inconsistent_new_component_pairs"]==2
