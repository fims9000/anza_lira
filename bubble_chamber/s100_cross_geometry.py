"""S100 unlabeled geometric-X candidates, not fiducial or particle truth.

S95 'CALIBRATION_CROSS_CONTEXT' only denotes proximity to supplied f1..f8
annotations; it does not autonomously detect calibration crosses. S100 looks
at S96's independently generated four-arm SOURCE geometry without using any
operator f1..f8. It can produce a *four-arm geometric X* suggestion for a
cross, a two-track overlap or an interaction; these are NOT distinguishable
by local 2D morphology alone.

Two source representations remain strictly separate:
- ORIGINAL_S86_COMPONENT_ENDPOINTS: S86/S87 observed component endpoints;
- VERIFIED_S86_MASK_SPLIT_ARMS: S96 alternative arms from frozen mask pixels.

No component is deleted, no same-particle identity assigned and no masked
center is called an observed bridge or true collision.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import acos, degrees, isfinite
from pathlib import Path

import numpy as np

SCHEMA="S100_UNLABELED_OBSERVED_2D_X_GEOMETRY_V1"
SOURCE_ORIGINAL="ORIGINAL_S86_COMPONENT_ENDPOINTS"
SOURCE_MASK="VERIFIED_S86_MASK_SPLIT_ARMS"
ACCEPT="FOUR_ARM_GEOMETRIC_X_REVIEW_NOT_FIDUCIAL_IDENTITY"
PAIRINGS=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))


@dataclass(frozen=True)
class XShapeConfig:
    max_opposite_deflection_deg:float=30.
    min_axes_cross_angle_deg:float=35.
    min_best_vs_second_pairing_margin_deg:float=10.
    max_center_to_port_px:float=14.
    max_cluster_diameter_px:float=22.
    max_junction_records:int=100000

    def validate(self):
        vals=(self.max_opposite_deflection_deg,
              self.min_axes_cross_angle_deg,
              self.min_best_vs_second_pairing_margin_deg,
              self.max_center_to_port_px,
              self.max_cluster_diameter_px)
        if (any(not isfinite(float(v)) for v in vals)
            or not 0<self.max_opposite_deflection_deg<=70
            or not 10<=self.min_axes_cross_angle_deg<=89
            or not 0<=self.min_best_vs_second_pairing_margin_deg<=75
            or not 2<=self.max_center_to_port_px<=48
            or not 4<=self.max_cluster_diameter_px<=96
            or type(self.max_junction_records) is not int
            or not 1<=self.max_junction_records<=500000):
            raise ValueError("Invalid bounded S100 geometric-X review parameters")


def _u(value):
    point=np.asarray(value,dtype=float)
    if point.shape!=(2,) or not np.all(np.isfinite(point)):
        raise ValueError("S100 port or junction center must be finite 2D")
    return point


def _unit(value):
    point=_u(value)
    norm=float(np.linalg.norm(point))
    if norm<1e-6 or abs(norm-1.)>.1:
        raise ValueError("S100 source outward tangent must be unit length")
    return point/norm


def _angle(a,b):
    return degrees(acos(float(np.clip(a@b,-1.,1.))))


def score_x_shape_junction(
    junction:dict,*,representation:str,
    source_bmp_sha256:str,
    cfg:XShapeConfig=XShapeConfig(),
):
    cfg.validate()
    if representation not in (SOURCE_ORIGINAL,SOURCE_MASK):
        raise ValueError("S100 source representation must be explicit")
    if (not isinstance(source_bmp_sha256,str)
        or len(source_bmp_sha256)!=64):
        raise ValueError("S100 source BMP SHA256 missing")
    try:int(source_bmp_sha256,16)
    except ValueError as e:raise ValueError("S100 source SHA invalid") from e
    ident=junction.get("junction_id")
    if not isinstance(ident,str) or not ident:
        raise ValueError("S100 source junction has missing ID")
    center=_u(junction.get("center_xy_source"))
    ports=junction.get("ports")
    if not isinstance(ports,list) or not ports:
        raise ValueError("S100 S96 junction must preserve original source ports")
    candidate_id="s100_x_"+sha256(
        (source_bmp_sha256+"|"+representation+"|"+ident).encode()
    ).hexdigest()[:24]
    result={
        "candidate_id":candidate_id,
        "source_s96_junction_id":ident,
        "source_representation":representation,
        "source_mask_tile_id":junction.get("source_mask_tile_id")
            if representation==SOURCE_MASK else None,
        "center_xy_source":[float(x) for x in center],
        "source_arm_count":len(ports),
        "source_S96_junction_status":junction.get("junction_status"),
        "cross_shape_review_status":None,
        "paired_source_arm_ids_proposed_visual_only":[],
        "max_opposite_turn_deflection_deg":None,
        "crossing_acute_axes_angle_deg":None,
        "best_vs_second_pairing_margin_deg":None,
        "original_components_preserved":True,
        "overlap_vs_interaction_vs_fiducial_undecidable":True,
        "not_automatic_calibration_fiducial_identity":True,
        "physical_auto_join_authorized":False,
        "model_only_not_independent_expert_truth":True,
    }
    if len(ports)!=4:
        result["cross_shape_review_status"]="NON_FOUR_ARM_ABSTAIN"
        return result
    ids=[x.get("component_id") for x in ports]
    keys=[(x.get("component_id"),x.get("port")) for x in ports]
    if (any(not isinstance(x,str) or not x for x in ids)
        or len(set(ids))!=4 or len(set(keys))!=4):
        result["cross_shape_review_status"]="DUPLICATE_SOURCE_ARM_ABSTAIN"
        return result
    vectors=[_unit(x.get("outward_unit")) for x in ports]
    positions=[_u(x.get("point_xy")) for x in ports]
    distances=[float(np.linalg.norm(p-center)) for p in positions]
    diameter=max(float(np.linalg.norm(a-b))
                 for i,a in enumerate(positions)
                 for b in positions[i+1:])
    if max(distances)>cfg.max_center_to_port_px:
        result["cross_shape_review_status"]="SOURCE_PORT_TOO_FAR_FROM_JUNCTION_ABSTAIN"
        return result
    if diameter>cfg.max_cluster_diameter_px or junction.get("initial_abstention"):
        result["cross_shape_review_status"]="SPREAD_OR_SOURCE_ABSTENTION"
        return result
    scored=[]
    for paired in PAIRINGS:
        bends=tuple(_angle(vectors[i],-vectors[j]) for i,j in paired)
        # Mean shape preference; also require EACH of two true-looking axes.
        scored.append((sum(bends),max(bends),paired,bends))
    scored.sort(key=lambda x:(x[0],x[1],x[2]))
    best,second=scored[0],scored[1]
    margin=second[0]-best[0]
    axes_angle=_angle(vectors[best[2][0][0]],vectors[best[2][1][0]])
    acute=min(axes_angle,180.-axes_angle)
    result["max_opposite_turn_deflection_deg"]=round(best[1],4)
    result["crossing_acute_axes_angle_deg"]=round(acute,4)
    result["best_vs_second_pairing_margin_deg"]=round(margin,4)
    if best[1]>cfg.max_opposite_deflection_deg:
        status="NO_TWO_STRAIGHT_OPPOSITE_AXES_ABSTAIN"
    elif acute<cfg.min_axes_cross_angle_deg:
        status="SHALLOW_AXIS_ANGLE_ABSTAIN"
    elif margin<cfg.min_best_vs_second_pairing_margin_deg:
        status="AMBIGUOUS_PAIRING_ABSTAIN"
    else:
        status=ACCEPT
        result["paired_source_arm_ids_proposed_visual_only"]=[
            [ids[i],ids[j]] for i,j in best[2]
        ]
    result["cross_shape_review_status"]=status
    return result


def build_x_shape_candidate_inventory(
    original_s96_junctions:list[dict],
    mask_derived_s96_junctions:list[dict],
    *,
    source_bmp_sha256:str,
    cfg:XShapeConfig=XShapeConfig(),
    near_overlap_radius_px:float=7.,
):
    """Keep BOTH representations; near matches are not physics deductions."""
    cfg.validate()
    total=len(original_s96_junctions)+len(mask_derived_s96_junctions)
    if total>cfg.max_junction_records or total<0:
        raise ValueError("S100 bounded geometry review exceeded")
    if not isfinite(near_overlap_radius_px) or not 1<=near_overlap_radius_px<=16:
        raise ValueError("S100 near-alternative-matching radius invalid")
    results=[]
    seen=set()
    for mode,source in ((SOURCE_ORIGINAL,original_s96_junctions),
                        (SOURCE_MASK,mask_derived_s96_junctions)):
        for row in source:
            item=score_x_shape_junction(
                row,representation=mode,source_bmp_sha256=source_bmp_sha256,cfg=cfg
            )
            if item["candidate_id"] in seen:
                raise ValueError("Duplicate S100 candidate from source S96 junction")
            seen.add(item["candidate_id"])
            results.append(item)
    originals=[x for x in results if
               x["source_representation"]==SOURCE_ORIGINAL
               and x["cross_shape_review_status"]==ACCEPT]
    for item in results:
        item["nearby_alternate_representation_candidates"]=[]
        if item["source_representation"]!=SOURCE_MASK or (
            item["cross_shape_review_status"]!=ACCEPT
        ):
            continue
        q=np.asarray(item["center_xy_source"],float)
        neighbors=sorted(
            (float(np.linalg.norm(q-np.asarray(other["center_xy_source"],float))),
             other["candidate_id"])
            for other in originals
            if np.linalg.norm(q-np.asarray(other["center_xy_source"],float))
            <=near_overlap_radius_px
        )
        item["nearby_alternate_representation_candidates"]=[
            {"candidate_id":ident,"center_distance_px":round(dist,4)}
            for dist,ident in neighbors[:4]
        ]
    from collections import Counter
    distribution=Counter(x["cross_shape_review_status"] for x in results)
    counts=Counter(x["source_representation"] for x in results)
    return results,{
        "schema":SCHEMA,
        "status":"MACHINE_GEOMETRIC_X_SHAPES_WITH_EXPLICIT_ABSTENTIONS",
        "source_bmp_sha256":source_bmp_sha256,
        "original_S86_endpoint_junctions_scored":counts[SOURCE_ORIGINAL],
        "alternative_verified_S86_mask_junctions_scored":counts[SOURCE_MASK],
        "total_machine_proposed_junctions_scored":total,
        "four_arm_geometric_X_review_candidates":distribution[ACCEPT],
        "reason_counts":dict(distribution),
        "original_alternative_representations_NOT_counted_as_distinct_physics_events":True,
        "near_matching_is_diagnostic_only_not_confirmed_identity":True,
        "calibration_cross_vs_track_overlap_vs_collision_undetermined":True,
        "no_unseen_source_pixels_claimed_observed":True,
        "no_GT_precision_recall_F1_computable":True,
        "thresholds_are_prior_defaults_not_real_H50Z_calibrated":{
            k:float(v) for k,v in vars(cfg).items() if k!="max_junction_records"
        },
    }
