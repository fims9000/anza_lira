"""S45 native registered-structure uncertainty, not a scratch detector or GT.

WARNING: requires both H50Z photographs. Its output is *transductive* and
MUST NOT be used as training labels for leave-one-photo-out experiments.
Never delete native candidates or turn 255 unknown into a training label.
"""
from __future__ import annotations

import cv2
import numpy as np

from .s39_machine_auto import ridge_evidence
from .s41_machine_quality import refine_pseudo_mask

SCHEMA = "s45_registered_repeat_uncertainty_transductive_v1"
MIN_COMMON_SUPPORT = 0.70
MIN_DISPLACED_MARGIN = 0.20


def repeat_advisory(evidence: dict) -> dict:
    """Frozen, uncalibrated conservative guard on *image* repeat support."""
    keys = ("source_multiscale_ridge_coverage",
            "counterpart_same_registered_path_coverage",
            "displaced_same_shape_counterpart_control_coverage")
    if not isinstance(evidence, dict) or any(k not in evidence for k in keys):
        raise ValueError("Missing S44 support provenance")
    a, b, ctrl = (evidence[k] for k in keys)
    if ctrl is None:
        return {"registered_repeat_risk": False,
                "reason": "NO_VALID_SHIFT_CONTROL", "native_particle_identity": "UNKNOWN"}
    if any(not isinstance(v, (int, float, np.number)) or
           not np.isfinite(v) or v < 0 or v > 1 for v in (a,b,ctrl)):
        raise ValueError("Uncalibrated S44 coverage out of range")
    if not isinstance(evidence.get("known_fiducial_uncertainty_NOT_NATIVE_TRUTH"), bool):
        raise ValueError("Known-fiducial flag missing")
    fid = evidence["known_fiducial_uncertainty_NOT_NATIVE_TRUTH"]
    repeated = (not fid and min(a,b) >= MIN_COMMON_SUPPORT and
                min(a,b) - ctrl >= MIN_DISPLACED_MARGIN)
    return {"registered_repeat_risk": bool(repeated),
            "reason": ("KNOWN_FIDUCIAL_SEPARATE_GUARD" if fid else
                       "STRONG_REGISTERED_REPEAT_NOT_ARTIFACT_TRUTH" if repeated else
                       "NO_RELIABLE_REGISTERED_REPEAT"),
            "native_particle_identity": "UNKNOWN",
            "requires_both_source_photos": True,
            "heldout_training_eligibility": "PROHIBITED_TRANSDUCTIVE_EVIDENCE"}


def _supported_ridge_mask(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Expected uint8 photo crop")
    scores = ridge_evidence(gray)["score"]
    threshold = float(np.percentile(scores, 95))
    if threshold <= 0:
        pos = scores[scores > 0]
        if not len(pos):
            return np.zeros(gray.shape, dtype=bool)
        threshold = float(np.quantile(pos, .3))
    return cv2.dilate(((scores >= threshold) & (scores > 0)).astype("uint8"),
                      np.ones((5,5),dtype="uint8")) > 0


def registered_uncertainty(gray: np.ndarray, other_registered: np.ndarray,
                           candidates: list[dict], evidence: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Mark only ridge-overlap pixels in selected line corridors as unknown.

    Do not label an entire line a scratch just because it repeats. This is
    inference-time machine annotation, never supervised/heldout ground truth.
    """
    if (gray.dtype != np.uint8 or other_registered.dtype != np.uint8 or
            gray.ndim != 2 or gray.shape != other_registered.shape or
            min(gray.shape) < 96 or len(candidates) != len(evidence)):
        raise ValueError("Wrong registered image and candidate provenance")
    decisions=[]
    lines=[]
    for c,e in zip(candidates,evidence):
        if c.get("draft_id") != e.get("draft_id"):
            raise ValueError("S39/S44 candidate identity or ordering mismatch")
        points=np.asarray(c.get("points_xy"),dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or
            not np.isfinite(points).all() or (points < 0).any() or
            (points[:,0]>=gray.shape[1]).any() or (points[:,1]>=gray.shape[0]).any()):
            raise ValueError("Out-of-frame original candidate")
        d=repeat_advisory(e)
        decisions.append({"draft_id":c["draft_id"], **d})
        if d["registered_repeat_risk"]:
            lines.append(np.rint(points).astype(np.int32))
    if not lines:
        return np.zeros(gray.shape,dtype=bool),decisions
    corridors=np.zeros(gray.shape,dtype=np.uint8)
    cv2.polylines(corridors,lines,False,1,7)
    paired=_supported_ridge_mask(gray) & _supported_ridge_mask(other_registered)
    protected=(corridors>0)&paired
    return protected,decisions


def downgrade_for_diagnostic_only(geometry_pseudo:np.ndarray, risk:np.ndarray)->tuple[np.ndarray,dict]:
    """Monotonic downgrade to unknown; specifically forbidden for heldout fit."""
    new,change=refine_pseudo_mask(geometry_pseudo,risk)
    change["transductive_other_photo_pixel_access"] = True
    change["leave_one_photo_out_model_training_allowed"] = False
    change["native_physics_ground_truth_created"] = 0
    return new,change


def graph_repeat_warning(graph:dict,decision_by_draft:dict[str,bool])->dict:
    """Annotate ALL relations; no physical bridge may be committed."""
    nodes=graph["nodes"]
    flags=[]
    for n in nodes:
        refs=n.get("source_candidate_ids")
        if not isinstance(refs,list) or not refs:
            raise ValueError("Missing original node provenance")
        if any(q not in decision_by_draft for q in refs):
            raise ValueError("Graph references unknown original proposal")
        flags.append(any(decision_by_draft[q] for q in refs))
    output=[]
    for e in graph["candidate_edges"]:
        i,j=e["i"],e["j"]
        if not (isinstance(i,int) and isinstance(j,int) and 0<=i<len(nodes)
                and 0<=j<len(nodes) and i!=j):
            raise ValueError("Invalid graph relation index")
        risk=bool(flags[i] or flags[j])
        output.append({"i":i,"j":j,
                       "status":"REGISTERED_REPEAT_ABSTAIN" if risk else
                                "UNVERIFIED_CANDIDATE_NO_AUTOJOIN",
                       "registered_repeat_endpoint_risk":risk,
                       "native_relation_truth":"UNKNOWN", "physical_auto_join_allowed":False})
    return {"node_count":len(nodes),"original_edge_count":len(output),
            "registered_repeat_node_count":sum(flags),
            "registered_repeat_edge_count":sum(r["registered_repeat_endpoint_risk"] for r in output),
            "physical_auto_joins_performed":0,"edges_PRIVATE":output}


def combine_with_known_fiducial_guard(s40_graph:dict, repeat_graph:dict)->dict:
    """Cross-tab S40 fiducial and S45 repeat risk without granting auto-joins."""
    before=s40_graph.get('candidate_edges')
    after=repeat_graph.get('edges_PRIVATE')
    if not isinstance(before,list) or not isinstance(after,list) or len(before)!=len(after):
        raise ValueError('Graph edges do not match S40 provenance')
    counters={k:0 for k in (
        'total','fiducial_risk','repeat_risk','overlap_both','new_repeat_not_fiducial',
        'union_any','unflagged_but_still_unverified','previously_ranked_unambiguous',
        'previously_ranked_unambiguous_union_risk')}
    for old,new in zip(before,after):
        if (old.get('i'),old.get('j'))!=(new.get('i'),new.get('j')):
            raise ValueError('Graph node edge order changed')
        if not isinstance(old.get('fiducial_overlap'),bool) or not isinstance(new.get('registered_repeat_endpoint_risk'),bool):
            raise ValueError('Missing trusted machine risk metadata')
        a=old['fiducial_overlap'];b=new['registered_repeat_endpoint_risk']
        if new['physical_auto_join_allowed'] is not False:
            raise ValueError('Physical joins must remain prohibited')
        u=bool(a or b)
        counters['total']+=1
        counters['fiducial_risk']+=int(a)
        counters['repeat_risk']+=int(b)
        counters['overlap_both']+=int(a and b)
        counters['new_repeat_not_fiducial']+=int(b and not a)
        counters['union_any']+=int(u)
        counters['unflagged_but_still_unverified']+=int(not u)
        ranked=bool(old.get('previously_ranked_unambiguous_UNVERIFIED',False))
        counters['previously_ranked_unambiguous']+=int(ranked)
        counters['previously_ranked_unambiguous_union_risk']+=int(ranked and u)
    counters['physical_auto_joins_performed']=0
    return counters
