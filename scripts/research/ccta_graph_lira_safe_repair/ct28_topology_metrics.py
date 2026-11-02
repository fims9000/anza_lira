"""GT-component topology audit for full-volume coronary repair evaluation.

Ground-truth segmentation is used only POST-HOC. We label each component of
the original broken mask by the reference component with largest voxel overlap.
For every pair of formerly disconnected original components that the output
mask merges, we then report whether those two components map to:
  * the same reference GT component (GT-consistent merge),
  * different reference GT components (GT-inconsistent merge), or
  * at least one unassigned GT component (unknown).

This is a *component-level* consistency check, not a guarantee against false
bridges *within* a connected coronary tree. It complements Dice/clDice.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from math import comb

import numpy as np
from scipy import ndimage


CONNECTIVITY=np.ones((3,3,3),dtype=np.uint8)


def gt_component_merge_audit(
    before:np.ndarray,after:np.ndarray,reference:np.ndarray,
    *,max_examples:int=25,
)->dict:
    if (before.ndim!=3 or before.shape!=after.shape or
        after.shape!=reference.shape or
        any(x.dtype!=bool for x in (before,after,reference))):
        raise ValueError("GT component audit requires aligned 3D bool masks")
    if np.any(before&~after):
        raise ValueError("After mask must preserve all input voxels")
    if max_examples<0:
        raise ValueError("max_examples must be >=0")
    b_labels,n_before=ndimage.label(before,structure=CONNECTIVITY)
    a_labels,n_after=ndimage.label(after,structure=CONNECTIVITY)
    gt_labels,n_gt=ndimage.label(reference,structure=CONNECTIVITY)
    owner=np.zeros(n_before+1,dtype=np.int32)
    support=np.zeros(n_before+1,dtype=np.float64)

    # Sparse overlap histogram of (old connected component, GT component).
    overlap=(b_labels>0)&(gt_labels>0)
    if overlap.any():
        codes=(
            b_labels[overlap].astype(np.int64)*(n_gt+1)
            +gt_labels[overlap].astype(np.int64)
        )
        found,counts=np.unique(codes,return_counts=True)
        for code,count in zip(found,counts):
            b=int(code//(n_gt+1))
            gt=int(code%(n_gt+1))
            if int(count)>support[b] or (int(count)==support[b] and gt<owner[b]):
                support[b]=int(count)
                owner[b]=gt

    # Pick one original voxel for each old component. Since the output mask
    # only adds voxels, all old voxels in that component must remain connected.
    nonzero=np.flatnonzero(b_labels.ravel())
    if nonzero.size:
        old=b_labels.ravel()[nonzero]
        unique,first=np.unique(old,return_index=True)
        parent=a_labels.ravel()[nonzero[first]]
        if np.any(parent<=0):
            raise RuntimeError("Output dropped a pre-existing component")
    else:
        unique=np.empty(0,dtype=int)
        parent=np.empty(0,dtype=int)
    group=defaultdict(list)
    for b,p in zip(unique,parent):
        group[int(p)].append(int(b))

    total=valid=false=unknown=0
    examples=[]
    for final_id,members in group.items():
        if len(members)<2:
            continue
        n=len(members)
        possible=comb(n,2)
        by_gt=Counter(int(owner[old_id]) for old_id in members)
        # Pairs with same nonzero GT ID; pairs with unknown either side
        # remain unknown, even if both are unassigned.
        good=sum(comb(cnt,2) for gt,cnt in by_gt.items() if gt>0 and cnt>=2)
        missing=by_gt.get(0,0)
        undecidable=comb(missing,2)+missing*(n-missing)
        bad=possible-good-undecidable
        if bad<0:
            raise RuntimeError("Inconsistent GT component pairing tally")
        total+=possible
        valid+=good
        false+=bad
        unknown+=undecidable
        if len(examples)<max_examples:
            examples.append({
                "after_component":final_id,
                "before_component_count":n,
                "before_component_ids":members[:25],
                "gt_component_owners":{
                    str(int(gt)):int(count)
                    for gt,count in sorted(by_gt.items())
                },
                "pair_count":possible,
                "gt_consistent_pairs":good,
                "gt_inconsistent_pairs":bad,
                "unassigned_pairs":undecidable,
            })

    return {
        "component_metric_scope":"posthoc_only_connected_components_26_neighborhood",
        "reference_component_count":int(n_gt),
        "original_component_count":int(n_before),
        "result_component_count":int(n_after),
        "original_components_with_GT_overlap":int(np.count_nonzero(owner[1:])),
        "original_components_without_GT_overlap":int(np.count_nonzero(owner[1:]==0)),
        "new_original_component_pairs_connected":int(total),
        "gt_consistent_new_component_pairs":int(valid),
        "gt_inconsistent_new_component_pairs":int(false),
        "gt_unassigned_new_component_pairs":int(unknown),
        "per_merge_examples":examples,
        "interpretation_warning":(
            "Same GT component does NOT imply anatomically correct branch connection; "
            "this metric cannot detect false bridges within an already connected tree."
        ),
    }
