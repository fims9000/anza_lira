"""S94: safe use of *sparse* H50Z keypoints in the S92 review pipeline.

'start' marks interaction candidates, not dense track pixels.
'f1'...'f8' marks calibration fiducials, not certified negative image masks.
The labels file contains NO verified centerlines, track identities or absence
labels. This module therefore only adjusts VIEW PRIORITIES and forces ABSTAIN
on candidate bridges near ambiguous operator keypoints; never deletes pixels.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import re

import numpy as np
from .s40_fiducial_safety import _min_distance_to_polyline

H50Z_RE=re.compile(r"^H50Z_\d{3}_\d{2}_\d{3}\.bmp$")
FID_RE=re.compile(r"^f[1-8]$")
SCHEMA="ANZA_S94_SPARSE_H50Z_KEYPOINT_GUARD_V1"


def _finite_number(value, name):
    if isinstance(value,bool):
        raise ValueError(name+" must be numeric")
    try:
        v=float(value)
    except (TypeError,ValueError):
        raise ValueError(name+" must be numeric") from None
    if not math.isfinite(v):
        raise ValueError(name+" must be finite")
    return v


def parse_photo_labels(raw:dict,photo:str,shape_yx:tuple[int,int]|list[int]):
    """Only an EXACT named original photo may receive its own keypoints."""
    if not isinstance(raw,dict) or not H50Z_RE.fullmatch(photo):
        raise ValueError("Expected original H50Z photo name and labels mapping")
    if len(shape_yx)!=2 or any(int(n)!=n or int(n)<=0 for n in shape_yx):
        raise ValueError("Invalid source image dimensions")
    h,w=map(int,shape_yx)
    if photo not in raw:
        return {"schema":SCHEMA,"status":"NO_MATCHING_PHOTO_LABELS",
                "photo":photo,"starts":[],"fiducials":[],
                "duplicate_fiducial_labels":[],
                "has_verified_track_masks":False}
    source=raw[photo]
    if not isinstance(source,dict) or not isinstance(source.get("keypoints"),list):
        raise ValueError("Malformed H50Z photo keypoints")
    iw=int(_finite_number(source.get("original_width"),"original_width"))
    ih=int(_finite_number(source.get("original_height"),"original_height"))
    if (ih,iw)!=(h,w):
        raise ValueError("Keypoint image dimensions do not match source BMP")
    points=source["keypoints"]
    expected=source.get("keypoints_count")
    if expected is not None and expected!=len(points):
        raise ValueError("Keypoint count mismatch")
    starts=[]
    fid=[]
    duplicate=set()
    seen={}
    for i,item in enumerate(points):
        if not isinstance(item,dict):
            raise ValueError("Invalid keypoint row")
        kind=item.get("label")
        if kind!="start" and not (isinstance(kind,str) and FID_RE.fullmatch(kind)):
            raise ValueError("Unknown H50Z keypoint type")
        x=_finite_number(item.get("x_px"),"x_px")
        y=_finite_number(item.get("y_px"),"y_px")
        if not (0<=x<w and 0<=y<h):
            raise ValueError("Keypoint coordinate outside source BMP")
        width=(None if item.get("width_px") is None
               else _finite_number(item["width_px"],"width_px"))
        if width is not None and not 0<width<=200:
            raise ValueError("Invalid point marker width")
        entry={"label":kind,"xy_source_px":[x,y],"row_index":i,
               "width_px":width}
        if kind=="start":
            starts.append(entry)
        else:
            if kind in seen:
                duplicate.add(kind)
            seen[kind]=i
            fid.append(entry)
    # If the same fiducial label was marked twice, its center is ambiguous:
    # never silently select one point or use either as a reliable guard.
    safe_fid=[x for x in fid if x["label"] not in duplicate]
    return {
        "schema":SCHEMA,
        "status":("DUPLICATE_FIDUCIAL_QUARANTINED" if duplicate else
                  "SPARSE_KEYPOINTS_ONLY_NOT_TRACK_GT"),
        "photo":photo,
        "starts":starts,"fiducials":safe_fid,
        "duplicate_fiducial_labels":sorted(duplicate),
        "has_verified_track_masks":False,
        "unlabeled_pixels_are_negative":False,
        "physical_track_identity_known":False,
    }


def audit_label_collection(raw:dict)->dict:
    """Validate a labels.json without opening even one BMP (private counts)."""
    if not isinstance(raw,dict) or not raw:
        raise ValueError("Empty or invalid labels.json")
    counts=Counter()
    duplicated=[]
    for photo,source in sorted(raw.items()):
        if not isinstance(source,dict):
            raise ValueError("Bad label record")
        h=int(_finite_number(source.get("original_height"),"height"))
        w=int(_finite_number(source.get("original_width"),"width"))
        row=parse_photo_labels(raw,photo,(h,w))
        counts["photos"]+=1
        counts["starts"]+=len(row["starts"])
        counts["usable_fiducials"]+=len(row["fiducials"])
        counts["quarantined_fiducial_marks"]+=sum(
            item.get("label") in row["duplicate_fiducial_labels"]
            for item in source["keypoints"]
        )
        if row["duplicate_fiducial_labels"]:
            duplicated.append({"photo":photo,
                               "fiducial_labels":row["duplicate_fiducial_labels"]})
    return {
        "schema":SCHEMA,"status":"COMPLETE_SPARSE_METADATA_AUDIT",
        **dict(counts),
        "duplicate_fiducial_cases":duplicated,
        "track_masks_present":False,
        "physical_class_labels_present":False,
        "unlabeled_pixels_are_background":False,
        "note":"Only clicked start/fiducial points; no particle track segmentation GT.",
    }


def _distance(p,point):
    return _min_distance_to_polyline(
        np.asarray(point,dtype=float),np.asarray(p,dtype=float)
    )


def _point_risks(path,context):
    start_near=[]
    fid_near=[]
    for row in context["starts"]:
        distance=_distance(path,row["xy_source_px"])
        radius=max(18.,min(36.,1.3*(row["width_px"] or 15.)))
        if distance<=radius:
            start_near.append(row["row_index"])
    for row in context["fiducials"]:
        distance=_distance(path,row["xy_source_px"])
        radius=max(40.,1.6*(row["width_px"] or 25.))
        if distance<=radius:
            fid_near.append(row["label"])
    return sorted(set(start_near)), sorted(set(fid_near))


def apply_sparse_keypoint_guard(components:list[dict],annotated:list[dict],
                                proposals:list[dict],selected:list[dict],
                                context:dict)->tuple[list[dict],dict]:
    """Perform AFTER S92 geometric ranking, BEFORE chain rescue/display export.

    Components and proposal original coordinates remain untouched. Keypoints
    can only increase caution or restore review visibility of tiny fragments.
    """
    if len(components)!=len(annotated):
        raise ValueError("Mismatch S92 observed components and review annotations")
    if context["status"]=="NO_MATCHING_PHOTO_LABELS":
        return selected,{
            "status":"NO_MATCHING_PHOTO_LABELS",
            "source_components_reviewed":len(components),
            "physical_auto_joins_authorized":False,
            "no_annotation_does_not_mean_no_tracks":True,
        }
    index={}
    near_start=set()
    risk_fid=set()
    for comp,item in zip(components,annotated):
        ident=comp["component_id"]
        if ident in index or ident!=item["component_id"]:
            raise ValueError("Unmatched or duplicate S92 component")
        path=np.asarray(comp["points_xy"],float)
        starts,fids=_point_risks(path,context)
        item["operator_start_nearby_row_indices_PRIVATE"]=starts
        item["fiducial_risk_labels_PRIVATE"]=fids
        item["keypoints_are_sparse_not_track_gt"]=True
        if starts:
            near_start.add(ident)
            item["review_flags"].append("NEAR_OPERATOR_START_NOT_A_TRACK_LABEL")
        if fids:
            risk_fid.add(ident)
            item["review_flags"].append("FIDUCIAL_MARK_REGION_AMBIGUOUS")
        index[ident]=path
    # Observe the exact proposed Hermite bridge AND endpoints; a fiducial X
    # near an inferred gap must never be silently merged into a particle.
    retained=[]
    abstentions=0
    for row in proposals:
        pa=row["component_a"]
        pb=row["component_b"]
        if pa not in index or pb not in index:
            raise ValueError("Unknown S92 proposal component")
        start_risk,mark_risk=_point_risks(
            row["inferred_bridge_points_xy"],context
        )
        # Even if the interior bridge misses the exact clicked point,
        # its source segments may still enter the calibration/vertex zone.
        if pa in near_start or pb in near_start:
            start_risk=sorted(set(start_risk)|{-1})
        if pa in risk_fid or pb in risk_fid:
            mark_risk=sorted(set(mark_risk)|{"segment_near_fiducial"})
        row["operator_start_zone_uncertain"]=bool(start_risk)
        row["fiducial_zone_uncertain"]=bool(mark_risk)
        if row["review_decision"]=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT" and (
            start_risk or mark_risk
        ):
            row["review_decision"]="KEYPOINT_REGION_ABSTAIN_NOT_A_TRACK_JOIN"
            abstentions+=1
        if row["review_decision"]=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
            retained.append(row)
    return retained,{
        "status":context["status"],
        "operator_starts":len(context["starts"]),
        "valid_fiducials":len(context["fiducials"]),
        "duplicate_fiducial_labels":context["duplicate_fiducial_labels"],
        "components_near_operator_start":len(near_start),
        "components_near_fiducial":len(risk_fid),
        "formerly_selected_gaps_abstained":abstentions,
        "remaining_geometric_gap_review_links":len(retained),
        "physical_auto_joins_authorized":False,
    }


def preserve_start_adjacent_short_marks(annotated:list[dict])->int:
    """S92 visibility override; does not relabel or delete one observed pixel."""
    restored=0
    for row in annotated:
        if (row.get("operator_start_nearby_row_indices_PRIVATE")
            and not row["show_in_primary_view"]):
            row["show_in_primary_view"]=True
            row["display_reason"]="NEAR_OPERATOR_START_MANUAL_REVIEW_NOT_GT"
            row["original_source_untouched"]=True
            restored+=1
    return restored
