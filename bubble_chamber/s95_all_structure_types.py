"""S95: classify ALL S86-observed bubble chamber line candidates for review.

A manual f1..f8 marker is a calibration-cross LOCATION, never a native-particle
mask. Track-like, artifact-like, coiling, and crowded crossings/bundles are
MACHINE-SUGGESTED morphology classes, not proven physical identities.
No source line is deleted. No same-particle join is authorized.

This module runs without any original BMP: it classifies frozen S86/S92
geometries, optionally source-locked sparse operator keypoints/curves.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import numpy as np
from scipy.spatial import cKDTree

from .s94_sparse_keypoint_guard import _point_risks
from .s95_operator_curves import operator_curve_support

SCHEMA="S95_ALL_OBSERVED_STRUCTURE_TYPES_V1"
PALETTE_BGR={
    "CALIBRATION_CROSS_CONTEXT":(55,125,255),
    "COILING_TRACE_CANDIDATE":(65,215,70),
    "BUNDLE_OR_INTERSECTION_REVIEW":(230,95,180),
    "TRACK_LIKE_CANDIDATE":(235,220,65),
    "ARTIFACT_LIKE_CANDIDATE":(105,105,235),
    "UNKNOWN":(195,195,195),
}
HUMAN_MARKER_CATEGORY="HUMAN_FIDUCIAL_MARK_LOCATION"


def _find_crowded_regions(components, *,radius_px:float=12.,min_neighbors:int=3,
                          max_support_points:int=300000):
    """Avoid treating several crossing filaments as one fitted parabola.

    A 'bundle' is an observed multi-fragment crowding hypothesis, not
    species, track identity or evidence of a neutron interaction.
    """
    if not 0<radius_px<=64 or not 2<=min_neighbors<=12:
        raise ValueError("Invalid crowded-region geometry")
    sample_points=[];owners=[];per_component=[]
    for idx,item in enumerate(components):
        p=np.asarray(item["points_xy"],float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
            raise ValueError("Bad S95 source component path")
        chosen=np.linspace(0,len(p)-1,min(len(p),40)).astype(int)
        source=p[chosen]
        per_component.append(source)
        sample_points.extend(source)
        owners.extend([idx]*len(source))
    if not sample_points:
        return {},{"status":"NO_COMPONENTS"}
    if len(sample_points)>max_support_points:
        return {},{"status":"DENSE_SOURCE_SUPPORT_UNEVALUATED","support_points":len(sample_points)}
    all_points=np.asarray(sample_points,float)
    all_owners=np.asarray(owners,dtype=np.int32)
    tree=cKDTree(all_points)
    result=defaultdict(list)
    # Query bounded number of representative centerline samples, including
    # interior points (crossings may not touch any fragment endpoint).
    for idx,support in enumerate(per_component):
        for q in support[::max(1,len(support)//9)]:
            neighborhood=tree.query_ball_point(q,radius_px)
            unique={int(all_owners[j]) for j in neighborhood}
            if len(unique)>=min_neighbors:
                result[idx].append((q.tolist(),sorted(unique)))
    summary={
        "status":"MACHINE_SPATIAL_CROWDING_ONLY",
        "support_points":len(sample_points),
        "components_with_crowding":len(result),
        "bundle_vs_intersection_unknown":True,
    }
    return result,summary


def type_all_observed_components(
    components:list[dict],
    annotated:list[dict],
    *,
    keypoint_context:dict|None=None,
    operator_curves:list[dict]|None=None,
    max_crowd_support:int=300000,
):
    """Full-coverage taxonomy of S86 centerlines AFTER S87 merging.

    Unlike S92 clutter view, this function deliberately NEVER drops short
    fragments. IDs must match source geometry exactly. Multi-label tags
    preserve 'spiral near a vertex' and 'track near fiducial' ambiguity.
    """
    if len(components)!=len(annotated):
        raise ValueError("S95 source component count mismatch")
    identifiers=[c["component_id"] for c in components]
    if len(identifiers)!=len(set(identifiers)):
        raise ValueError("Duplicate observed source component")
    if any(c["component_id"]!=a["component_id"] for c,a in zip(components,annotated)):
        raise ValueError("S95 source identity order mismatch")
    if operator_curves is None:
        operator_curves=[]
    if keypoint_context is not None and keypoint_context.get("status")=="NO_MATCHING_PHOTO_LABELS":
        keypoint_context=None
    crowded,crowding_summary=_find_crowded_regions(
        components,max_support_points=max_crowd_support,
    )
    rows=[]
    for i,(c,shape) in enumerate(zip(components,annotated)):
        source=np.asarray(c["points_xy"],float)
        fids=[];starts=[]
        if keypoint_context is not None:
            starts,fids=_point_risks(source,keypoint_context)
        refs=operator_curve_support(source,operator_curves)
        coil=shape["shape_review_category"]=="COILING_TRACE_REVIEW"
        artifact=shape["shape_review_category"]=="LOW_PRIORITY_POSSIBLE_ARTIFACT"
        crowd=i in crowded
        # The operator's explicit track geometry is an inspected guide but
        # not proof every measured pixel belongs to a real particle.
        best_ref=refs["status"]=="NEAR_SINGLE_OPERATOR_GUIDE_REVIEW"
        if fids:
            klass="CALIBRATION_CROSS_CONTEXT"
        elif crowd:
            klass="BUNDLE_OR_INTERSECTION_REVIEW"
        elif coil:
            klass="COILING_TRACE_CANDIDATE"
        elif best_ref or not artifact:
            klass="TRACK_LIKE_CANDIDATE"
        elif artifact:
            klass="ARTIFACT_LIKE_CANDIDATE"
        else:
            klass="UNKNOWN"
        tags=[]
        if fids:tags.append("NEAR_HUMAN_CALIBRATION_CROSS")
        if starts:tags.append("NEAR_HUMAN_OPERATOR_VERTEX")
        if crowd:tags.append("MULTIPLE_IMAGE_FRAGMENTS_MEET_OR_CROSS")
        if coil:tags.append("COILED_2D_SHAPE_NO_CAUSE_IDENTIFIED")
        if artifact:tags.append("SHORT_ANGULAR_OR_HAIRPIN_MAY_BE_TRUE_TRACK")
        if best_ref:tags.append("NEAR_SOURCE_LOCKED_OPERATOR_CURVE_GUIDE")
        if refs["status"]=="AMBIGUOUS_MULTIPLE_OPERATOR_GUIDES":
            tags.append("AMBIGUOUS_OPERATOR_TRACK_IDENTITY")
        row={
            "component_id":c["component_id"],
            "source_fragment_ids":list(c["source_fragment_ids"]),
            "primary_review_category":klass,
            "all_morphology_tags":tags,
            "operator_guide_support":refs,
            "fiducial_labels_nearby_PRIVATE":fids,
            "operator_start_row_indices_nearby_PRIVATE":starts,
            "crowding_hypotheses":[{"point_xy_source":pt,"neighbor_components":len(other)}
                                   for pt,other in crowded.get(i,[])[:3]],
            "shape_observed_length_px":shape["observed_length_px"],
            "shape_signed_turn_deg":shape["total_signed_turn_deg"],
            "suggested_visible_in_main_view":bool(shape.get("show_in_primary_view",True)),
            "preserved_source_observed":True,
            "machine_only":True,
            "physical_track_identity":"UNKNOWN",
            "could_be_other_track_at_crossing":True,
            "physical_auto_join_authorized":False,
        }
        rows.append(row)
    categories=dict(Counter(x["primary_review_category"] for x in rows))
    return rows,{
        "schema":SCHEMA,"status":"ALL_EXTRACTED_SOURCE_COMPONENTS_TYPED",
        "extracted_components":len(rows),
        "typed_components":len(rows),
        "categories":categories,
        "crowding":crowding_summary,
        "not_all_visible_film_lines_guaranteed_detected":True,
        "cannot_certify_garbage_without_review":True,
        "no_source_components_deleted":True,
        "physical_auto_joins_authorized":False,
        "operator_guides_not_dense_pixel_gt":True,
        "class_semantics":{
            "CALIBRATION_CROSS_CONTEXT":"Observed fragment near named human f1..f8 mark; not proof entire fragment is a cross",
            "TRACK_LIKE_CANDIDATE":"Extended or supported line candidate; particle not verified",
            "ARTIFACT_LIKE_CANDIDATE":"Short/angular/hairpin visual artifact; could be a real short track",
            "BUNDLE_OR_INTERSECTION_REVIEW":"Crowded line neighborhood; genuine bundle vs crossing unresolved",
            "COILING_TRACE_CANDIDATE":"Large coherent image-plane turn; collision mechanism unknown",
            "UNKNOWN":"Unresolved visual class",
        }
    }


def human_mark_objects(keypoint_context:dict|None)->list[dict]:
    """Operator-mark LOCATION objects, separate from predicted source lines."""
    if keypoint_context is None or keypoint_context.get("status")=="NO_MATCHING_PHOTO_LABELS":
        return []
    marks=[{
        "object_id":f"S95_HUMAN_FID_{row['label']}_{row['row_index']}",
        "category":HUMAN_MARKER_CATEGORY,
        "position_xy":row["xy_source_px"],
        "marker_label":row["label"],
        "annotation_source":"H50Z_SPARSE_KEYPOINT_OPERATOR",
        "underlying_detected_image_mark_verified":False,
        "not_a_particle_track":True,
    } for row in keypoint_context["fiducials"]]
    marks.extend({
        "object_id":f"S95_HUMAN_START_{row['row_index']}",
        "category":"HUMAN_OPERATOR_VERTEX_LOCATION",
        "position_xy":row["xy_source_px"],
        "marker_label":"start",
        "annotation_source":"H50Z_SPARSE_KEYPOINT_OPERATOR",
        "underlying_detected_track_verified":False,
    } for row in keypoint_context["starts"])
    return marks
