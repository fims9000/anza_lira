"""S97: conditional visual-instance consistency of S92/S96 proposed 2D links.

Human original-photo strokes have explicit *visual instance* IDs, independent
of candidate model output. Match observed S86 components to distinct visible
strokes conservatively, then inspect whether a proposed join spans two
different independently labeled visible lines.

This is NOT physical particle accuracy or complete candidate recall:
  * a mixed source component or unlabeled source point ABSTAINS;
  * two human-drawn instances may overlap in projection, so even consensus
    of the image can be physically ambiguous at a crossing;
  * no unproposed link is automatically a false negative.
"""
from __future__ import annotations

from collections import Counter,defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .s97_manual_reference import _point_array,_resample_polyline,KINDS,CERTAINTY

SCHEMA="S97_PARTIAL_MANUAL_VISUAL_INSTANCE_LINK_CONSISTENCY_V1"


def map_observed_components_to_reference(
    human_strokes:list[dict],
    observed_components:list[dict],
    *,
    radius_px:float=2.,
    min_fraction:float=.65,
    minimum_margin:float=.20,
):
    if not (0.5<=radius_px<=8. and .4<=min_fraction<=1.
            and .05<=minimum_margin<=.5):
        raise ValueError("S97 unsafe model-to-human matching thresholds")
    coords=[];labels=[]
    for row in human_strokes:
        if row["certainty"]!="CLEAR" or row["kind"]=="UNKNOWN":
            continue
        sampled,_=_resample_polyline(_point_array(row["points_xy_source_px"]))
        if len(coords)+len(sampled)>2000000:
            raise ValueError("Too many full-resolution human reference points")
        coords.extend(sampled)
        labels.extend([(row["visual_instance_id"],row["kind"])]*len(sampled))
    if not coords:
        return {x["component_id"]:None for x in observed_components},{
            "source_components":len(observed_components),
            "matched_components":0,
            "status":"NO_CLEAR_HUMAN_REFERENCE"
        }
    samples=np.asarray(coords)
    tree=cKDTree(samples)
    labels=np.asarray(labels,dtype=object)
    lookup={}
    stats=Counter()
    for row in observed_components:
        ident=row["component_id"]
        if ident in lookup:
            raise ValueError("Duplicate S97 component ID in link matching")
        curve=_point_array(row["points_xy"],max_points=1000000)
        chosen=curve[np.linspace(0,len(curve)-1,min(64,len(curve))).astype(int)]
        dist,idx=tree.query(chosen,k=1,workers=1)
        found=dist<=radius_px+1e-9
        totals=Counter(labels[int(index)][0] for index,m in zip(idx,found) if m)
        if not totals:
            stats["UNANNOTATED_OR_OUTSIDE_MATCH_RADIUS"]+=1
            lookup[ident]=None
            continue
        ordered=totals.most_common()
        winner,winner_n=ordered[0]
        second_n=ordered[1][1] if len(ordered)>1 else 0
        matched_fraction=winner_n/len(chosen)
        margin=(winner_n-second_n)/len(chosen)
        if matched_fraction<min_fraction or margin<minimum_margin:
            stats["AMBIGUOUS_OR_MIXED_VISUAL_INSTANCES"]+=1
            lookup[ident]=None
            continue
        # If one human ID has contradictory kinds, refuse automatic label.
        human_kinds={labels[int(index)][1] for index,m in zip(idx,found)
                     if m and labels[int(index)][0]==winner}
        if len(human_kinds)!=1:
            stats["INCONSISTENT_HUMAN_CLASS_WITHIN_VISUAL_INSTANCE"]+=1
            lookup[ident]=None
            continue
        result={
            "visual_instance_id":str(winner),
            "human_kind":str(next(iter(human_kinds))),
            "observed_component_support_fraction":round(matched_fraction,5),
            "dominance_margin_fraction":round(margin,5),
            "physical_identity":"NOT_ESTABLISHED",
        }
        lookup[ident]=result
        stats["MATCHED_CLEAR_VISUAL_INSTANCE"]+=1
    return lookup,{
        "source_components":len(observed_components),
        "matched_components":stats["MATCHED_CLEAR_VISUAL_INSTANCE"],
        "mapping_status_counts":dict(stats),
        "unmatched_not_negative":True,
        "match_based_on_source_original_pixels_not_machine_proposals":True,
    }


def audit_provisional_joins(*,edges:list[dict],component_labels:dict,
                            source_kind:str):
    if source_kind not in ("S92_GAP","S96_CROSSING"):
        raise ValueError("Unsupported same-photo provisional visual join type")
    expected_decision={
        "S92_GAP":"PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
        "S96_CROSSING":"PROVISIONAL_2D_CROSSING_CONTINUATION",
    }[source_kind]
    selected=[]
    n_candidates=0
    for e in edges:
        n_candidates+=1
        if e.get("status")!="UNKNOWN" or e.get("physical_auto_join") is not False:
            raise ValueError("S97 audit refuses physically asserted machine links")
        if e.get("review_decision")!=expected_decision:
            continue
        a,b=e["component_a"],e["component_b"]
        if a not in component_labels or b not in component_labels:
            raise ValueError("Join references a non-source S86 observed component")
        label_a=component_labels[a]
        label_b=component_labels[b]
        if label_a is None or label_b is None:
            outcome="ABSTAIN_INCOMPLETE_HUMAN_SOURCE_COVERAGE"
        elif (label_a["visual_instance_id"]==label_b["visual_instance_id"]
              and label_a["human_kind"]==label_b["human_kind"]):
            outcome="CONSISTENT_WITH_ANNOTATED_VISUAL_INSTANCE"
        else:
            outcome="CONTRADICTS_ANNOTATED_VISUAL_INSTANCES"
        selected.append({
            "component_a":a,"component_b":b,
            "human_visual_evidence_status":outcome,
            "is_image_level_not_physical_identity":True,
        })
    counts=Counter(x["human_visual_evidence_status"] for x in selected)
    consistent=counts["CONSISTENT_WITH_ANNOTATED_VISUAL_INSTANCE"]
    contrary=counts["CONTRADICTS_ANNOTATED_VISUAL_INSTANCES"]
    return {
        "schema":SCHEMA,
        "status":"CONDITIONAL_INDEPENDENT_HUMAN_2D_VISUAL_INSTANCE_CHECK",
        "machine_source":source_kind,
        "all_geometry_candidate_edges":n_candidates,
        "selected_review_level_links":len(selected),
        "assessable_source_annotated_links":consistent+contrary,
        "consistent_with_independent_human_2d_instances":consistent,
        "contradicts_independent_human_2d_instances":contrary,
        "abstained_unreviewable_links":counts["ABSTAIN_INCOMPLETE_HUMAN_SOURCE_COVERAGE"],
        "conditional_disagreement_fraction":(
            contrary/(consistent+contrary) if consistent+contrary else None
        ),
        "selected_link_outcomes":selected,
        "full_unproposed_pair_recall_not_measured":True,
        "not_a_precision_f1_or_particle_identity_measure":True,
        "inferred_gap_pixels_never_count_as_observed_centerline":True,
    }
