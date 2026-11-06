"""S98: visual-trace fragmentation and *reference-first* detected-piece continuity.

S97 marks ONLY human-visible 2D curves on the ORIGINAL BMP (no model UI).
S98 subsequently maps all S86/S87 observed source centerlines onto those
human-positive curves. It discovers boundaries between distinct detected
pieces by walking the fixed human polyline in arclength order, INDEPENDENT
of S92/S96 proposed joins, and checks whether either method suggests them.

A detected-piece adjacency is not a known same-PARTICLE ground-truth edge:
it is a SAME-ANNOTATED-VISUAL-TRACE boundary conditional on both endpoint
fragments being sufficiently observed. Unobserved whole tracks, rare
ambiguous junctions, and unlabeled image background cannot be evaluated.

Never treat bridge interpolations as observed source pixels. Never mutate
S86/S92/S96 or physically auto-connect trajectories.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from hashlib import sha256
from math import isfinite

import numpy as np
from scipy.spatial import cKDTree

from .s97_manual_reference import (
    POSITIVE_KINDS,_point_array,_resample_polyline
)

SCHEMA="S98_INDEPENDENT_VISUAL_CONTINUITY_BOUNDARIES_V1"
MAX_SOURCE_POINTS=2_000_000
MAX_STROKE_SAMPLES=200_000


def _observed_index(components:list[dict]):
    """Index original S86 observed polyline points, retaining each source ID."""
    ids=set()
    coords=[]
    tags=[]
    count=0
    for item in components:
        cid=item["component_id"]
        if not isinstance(cid,str) or not cid or cid in ids:
            raise ValueError("S98 original source component ID duplicated/missing")
        ids.add(cid)
        pts=_point_array(item["points_xy"],min_points=2,max_points=1_000_000)
        count+=len(pts)
        if count>MAX_SOURCE_POINTS:
            raise ValueError("S98 original source exceeds explicit point budget")
        coords.append(pts)
        tags.extend([cid]*len(pts))
    if not coords:
        return None,np.empty(0,dtype=object),ids
    return cKDTree(np.concatenate(coords)),np.asarray(tags,dtype=object),ids


def _sample_assignment(points,tree,tags,*,radius_px,ambiguity_margin_px):
    """Assign expert samples to a nearby original component, or abstain.

    The 12-nearest observed-point search identifies competing distinct IDs.
    Where a different component is nearly as close, assigning a physical/
    visual identity from a simple nearest point is unsafe: mark ambiguous.
    If all 12 neighbors belong to one component we keep it as tentative
    *observed source* assignment, not proof there was no other faint line.
    """
    if tree is None:
        return [None]*len(points),np.zeros(len(points),dtype=bool)
    neighbors=min(12,len(tags))
    distances,positions=tree.query(points,k=neighbors,workers=1)
    if neighbors==1:
        distances=distances.reshape(-1,1)
        positions=positions.reshape(-1,1)
    output=[]
    ambiguity=np.zeros(len(points),dtype=bool)
    for i,(dd,ii) in enumerate(zip(distances,positions)):
        if float(dd[0])>radius_px:
            output.append(None)
            continue
        first=str(tags[int(ii[0])])
        rival=False
        for value,index in zip(dd[1:],ii[1:]):
            if float(value)>radius_px:
                break
            if str(tags[int(index)])!=first and float(value)-float(dd[0])<=ambiguity_margin_px:
                rival=True
                break
        if rival:
            output.append(None)
            ambiguity[i]=True
        else:
            output.append(first)
    return output,ambiguity


def _runs(seq):
    if not seq:
        return []
    segments=[]
    begin=0
    for i in range(1,len(seq)+1):
        if i==len(seq) or seq[i]!=seq[begin]:
            segments.append({
                "component_id":seq[begin],
                "begin":begin,"end":i-1,"support_samples":i-begin
            })
            begin=i
    return segments


def discover_human_trace_piece_boundaries(
    *,
    human_strokes:list[dict],
    observed_components:list[dict],
    radius_px:float=2.,
    source_sampling_step_px:float=1.,
    ambiguity_margin_px:float=.45,
    min_run_support_px:float=3.,
    max_unobserved_gap_along_trace_px:float=80.,
):
    """Find S86 piece transitions along explicit, CLEAR human-positive lines.

    A potential positive machine-piece boundary is generated from ORIGINAL
    expert strokes; no S92/S96 links, scores or model-selected examples used.
    No claim of any link-recall denominator beyond these detected pieces.
    """
    numeric=(radius_px,source_sampling_step_px,ambiguity_margin_px,
             min_run_support_px,max_unobserved_gap_along_trace_px)
    if (not all(isfinite(x) for x in numeric)
        or not (.5<=radius_px<=8.)
        or not (.5<=source_sampling_step_px<=2.)
        or not (.05<=ambiguity_margin_px<=2.)
        or not (2.<=min_run_support_px<=20.)
        or not (5.<=max_unobserved_gap_along_trace_px<=500.)):
        raise ValueError("S98 unsafe/invalid fixed distance thresholds")
    tree,tags,source_ids=_observed_index(observed_components)
    report=[]
    boundary_index={}
    sample_total=ambiguous_total=0
    classified_instances=defaultdict(set)
    seen_strokes=set()
    for human in human_strokes:
        sid=human["stroke_id"]
        if sid in seen_strokes:
            raise ValueError("Duplicate independent expert reference stroke")
        seen_strokes.add(sid)
        if human["kind"] not in POSITIVE_KINDS or human["certainty"]!="CLEAR":
            continue
        visual_id=human["visual_instance_id"]
        human_points=_point_array(human["points_xy_source_px"])
        samples,length=_resample_polyline(human_points,step_px=source_sampling_step_px)
        if len(samples)>MAX_STROKE_SAMPLES:
            raise ValueError("S98 human trace exceeds source sample budget")
        gap_unit=length/(len(samples)-1)
        assigned,ambiguous=_sample_assignment(
            samples,tree,tags,radius_px=radius_px,
            ambiguity_margin_px=ambiguity_margin_px
        )
        n_min=int(np.ceil(min_run_support_px/gap_unit))
        valid_runs=[
            r for r in _runs(assigned)
            if r["component_id"] is not None and r["support_samples"]>=n_min
        ]
        piece_ids=sorted({r["component_id"] for r in valid_runs})
        for cid in piece_ids:
            classified_instances[cid].add(visual_id)
        adjacency=[]
        for left,right in zip(valid_runs,valid_runs[1:]):
            if left["component_id"]==right["component_id"]:
                continue
            # Refuse to cross an intervening stable third source component.
            # Every run of >= n_min samples survived, so consecutive stable
            # runs represent consecutive visual evidence on this SAME curve.
            gap_steps=max(0,right["begin"]-left["end"]-1)
            gap_px=round(gap_steps*gap_unit,4)
            if gap_px>max_unobserved_gap_along_trace_px:
                continue
            a,b=sorted((left["component_id"],right["component_id"]))
            key=(a,b)
            pair_id="s98_adj_"+sha256((a+"|"+b).encode()).hexdigest()[:20]
            option={
                "pair_id":pair_id,
                "component_a":a,"component_b":b,
                "human_visual_stroke_id":sid,
                "human_visual_instance_id":visual_id,
                "reference_intermediate_gap_length_px":gap_px,
                "human_stroke_sample_index_from":left["end"],
                "human_stroke_sample_index_to":right["begin"],
                "support_samples_before":left["support_samples"],
                "support_samples_after":right["support_samples"],
                "human_positive_visual_continuity_only":True,
                "no_particle_identity_or_physical_auto_join":True,
            }
            adjacency.append(option)
            if key not in boundary_index:
                boundary_index[key]={
                    "pair_id":pair_id,
                    "component_a":a,"component_b":b,
                    "supporting_human_stroke_ids":set(),
                    "supporting_human_visual_instance_ids":set(),
                    "max_reference_intermediate_gap_px":0.,
                    "min_reference_intermediate_gap_px":float("inf"),
                    "sampled_visual_evidence_occurrences":0,
                }
            slot=boundary_index[key]
            slot["supporting_human_stroke_ids"].add(sid)
            slot["supporting_human_visual_instance_ids"].add(visual_id)
            slot["max_reference_intermediate_gap_px"]=max(
                slot["max_reference_intermediate_gap_px"],gap_px
            )
            slot["min_reference_intermediate_gap_px"]=min(
                slot["min_reference_intermediate_gap_px"],gap_px
            )
            slot["sampled_visual_evidence_occurrences"]+=1
        ambiguous_total+=int(ambiguous.sum())
        sample_total+=len(samples)
        report.append({
            "stroke_id":sid,"visual_instance_id":visual_id,
            "human_visible_type":human["kind"],
            "human_curve_length_px":round(length,4),
            "human_sample_count":len(samples),
            "nearest_source_assignment_ambiguous_samples":int(ambiguous.sum()),
            "stable_observed_component_runs":len(valid_runs),
            "unique_stable_observed_component_ids":piece_ids,
            "unique_stable_detected_pieces":len(piece_ids),
            "provisional_visual_adjacencies":adjacency,
            "provisional_adjacencies_count":len(adjacency),
            "note":"fragmentation among observed S86 components, not independent physical particle count",
        })

    # A single ORIGINAL S86 component that strongly maps to TWO separately
    # annotated visible instances is an ambiguous fused/crossing component.
    # Do not call its pairing correct or incorrect without human review.
    mixed={cid for cid,ids in classified_instances.items() if len(ids)>1}
    boundaries=[]
    ambiguous_boundaries=[]
    for key,slot in sorted(boundary_index.items()):
        compatible=len(slot["supporting_human_visual_instance_ids"])==1
        safe=compatible and key[0] not in mixed and key[1] not in mixed
        entry={
            **slot,
            "supporting_human_stroke_ids":sorted(slot["supporting_human_stroke_ids"]),
            "supporting_human_visual_instance_ids":sorted(slot["supporting_human_visual_instance_ids"]),
            "max_reference_intermediate_gap_px":round(slot["max_reference_intermediate_gap_px"],4),
            "min_reference_intermediate_gap_px":round(slot["min_reference_intermediate_gap_px"],4),
            "visual_continuity_conditional_on_human_strokes":True,
            "physical_same_particle_verification":False,
        }
        if safe:
            entry["evidence_status"]="VISIBLE_SAME_STROKE_ADJACENCY_SOURCE_PIECES"
            boundaries.append(entry)
        else:
            entry["evidence_status"]="ABSTAIN_MIXED_VISUAL_INSTANCES_OR_FUSED_SOURCE_COMPONENT"
            ambiguous_boundaries.append(entry)
    return {
        "schema":SCHEMA,
        "status":"INDEPENDENT_HUMAN_FIRST_CONDITIONAL_SOURCE_PIECE_BOUNDARIES",
        "source_observed_component_count":len(source_ids),
        "human_clear_positive_strokes":len(report),
        "human_positive_stroke_samples":sample_total,
        "human_samples_with_tied_source_assignment":ambiguous_total,
        "source_components_touching_multiple_human_visual_instances":sorted(mixed),
        "assessable_unique_detected_piece_adjacencies":len(boundaries),
        "ambiguous_unique_piece_adjacencies_abstained":len(ambiguous_boundaries),
        "per_human_stroke":report,
        "conditionally_assessable_piece_adjacencies":boundaries,
        "ambiguous_piece_adjacencies":ambiguous_boundaries,
        "reference_guided_source_component_assignment_not_physics_GT":True,
        "full_unobserved_track_or_unproposed_physical_link_recall_not_measurable":True,
        "no_unannotated_pixels_assigned_negative_labels":True,
        "source_S86_S92_S96_unchanged":True,
        "source_interpolated_gaps_not_observed_pixels":True,
        "fixed_method_parameters":{
            "radius_px":radius_px,
            "source_sampling_step_px":source_sampling_step_px,
            "ambiguity_margin_px":ambiguity_margin_px,
            "min_run_support_px":min_run_support_px,
            "max_unobserved_gap_along_trace_px":max_unobserved_gap_along_trace_px,
        },
    }


def measure_proposal_recovery(
    reference_first:dict,
    *,
    proposal_sets:dict[str,list[dict]]
):
    """Evaluate S92/S96 *selected* visual links against reference-first pairs.

    The denominator is formed WITHOUT inspecting proposals, and is conditional
    on both source components appearing as distinct observed human-positive
    pieces. Therefore this is NOT full truth recall of all physical links.
    """
    if reference_first.get("schema")!=SCHEMA:
        raise ValueError("Unknown S98 independent first source")
    truth={
        (x["component_a"],x["component_b"]):x
        for x in reference_first["conditionally_assessable_piece_adjacencies"]
    }
    if len(truth)!=len(reference_first["conditionally_assessable_piece_adjacencies"]):
        raise ValueError("Repeated S98 independent source adjacency")
    source_ids={
        cid for stroke in reference_first["per_human_stroke"]
        for cid in stroke["unique_stable_observed_component_ids"]
    }
    source_ids.update(reference_first["source_components_touching_multiple_human_visual_instances"])
    all_metrics=[]
    selected_unions=set()
    by_method={}
    for name,rows in proposal_sets.items():
        if name not in ("S92_GAP","S96_CROSSING") or not isinstance(rows,list):
            raise ValueError("Unsupported or malformed S98 machine proposal type")
        selected=set()
        proposed=set()
        for p in rows:
            if p.get("physical_auto_join") is not False or p.get("status")!="UNKNOWN":
                raise ValueError("S98 refuses machine links claiming physical truth")
            a,b=p["component_a"],p["component_b"]
            if a==b:
                raise ValueError("Self-join rejected")
            # An edge may legitimately touch an unannotated source piece.
            # Only source membership of *matching labels* is assessed below;
            # unrelated unknown components are not certified negative.
            pair=tuple(sorted((a,b)))
            proposed.add(pair)
            if p.get("review_decision")=={
                "S92_GAP":"PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
                "S96_CROSSING":"PROVISIONAL_2D_CROSSING_CONTINUATION",
            }[name]:
                selected.add(pair)
        selected_unions.update(selected)
        denom=len(truth)
        count_proposed=sum(key in proposed for key in truth)
        count_selected=sum(key in selected for key in truth)
        method={
            "source_method":name,
            "source_all_proposed_unique_pair_count":len(proposed),
            "source_selected_unique_pair_count":len(selected),
            "reference_first_unique_observed_piece_adjacencies":denom,
            "proposed_same_human_stroke_reference_boundaries":count_proposed,
            "selected_same_human_stroke_reference_boundaries":count_selected,
            "conditional_reference_piece_boundary_proposal_coverage":(
                count_proposed/denom if denom else None
            ),
            "conditional_reference_piece_boundary_selected_coverage":(
                count_selected/denom if denom else None
            ),
            "reference_boundaries_not_selected":[
                truth[key]["pair_id"] for key in sorted(truth)
                if key not in selected
            ],
            "not_full_true_track_link_recall":True,
        }
        by_method[name]=method
        all_metrics.append(method)
    denom=len(truth)
    return {
        "schema":SCHEMA,
        "status":"S98_MATCHED_REFERENCE_FIRST_SOURCE_PIECE_BOUNDARY_COVERAGE",
        "reference_first_unique_source_piece_boundaries":denom,
        "models_compared":all_metrics,
        "union_of_selected_candidate_pairs_covers_reference_boundaries":(
            sum(key in selected_unions for key in truth)
        ),
        "union_selected_conditional_coverage":(
            sum(key in selected_unions for key in truth)/denom if denom else None
        ),
        "reference_boundaries_missed_by_all_selected_models":[
            truth[key] for key in sorted(truth) if key not in selected_unions
        ],
        "partial_human_visible_not_particle_GT":True,
        "does_not_estimate_unseen_weak_undetected_track_recall":True,
        "does_not_estimate_false_positive_rate_without_exhaustive_background_GT":True,
    }
