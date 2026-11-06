"""S97: independent, PARTIAL positive-only human 2-D centerline references.

Annotations are drawn over ORIGINAL BMP pixels with no machine overlay.
They are image-level expert reference strokes, NOT particle/interaction GT.

Consequently:
  * Recall/coverage is conditional on expressly drawn, clear visual traces.
  * A machine pixel outside an annotated stroke is NOT a false positive.
  * Missing expert strokes and tiles not reviewed are NOT certified negatives.
  * Inferred S92/S96 bridges are never counted as observed S86 centerlines.
  * A single image or multiple same-photo crops do not give an independent
    held-out experiment or valid patient/film-bootstrap confidence interval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree

SCHEMA="S97_INDEPENDENT_HUMAN_2D_STROKE_REFERENCE_V1"
KINDS=(
    "TRACK_LIKE","COILING_TRACK_LIKE","FIDUCIAL_CROSS",
    "ARTIFACT","CROSSING_OR_BUNDLE","UNKNOWN",
)
CERTAINTY=("CLEAR","UNCERTAIN")
POSITIVE_KINDS=frozenset(("TRACK_LIKE","COILING_TRACK_LIKE"))
EVALUATION_SCHEMA="S97_PARTIAL_POSITIVE_CENTERLINE_COVERAGE_V1"


def _positive_integer(value,what):
    if type(value) is not int or value<=0:
        raise ValueError(f"Invalid positive integer {what}")
    return value


def _point_array(value,*,min_points=2,max_points=20000):
    v=np.asarray(value,dtype=float)
    if (v.ndim!=2 or v.shape[1]!=2 or not min_points<=len(v)<=max_points
        or not np.isfinite(v).all()):
        raise ValueError("S97 finite 2D stroke points required")
    if np.any(np.linalg.norm(np.diff(v,axis=0),axis=1)<1e-6):
        raise ValueError("Repeated adjacent S97 operator stroke points")
    return v


def _resample_polyline(p,*,step_px=1.,max_length_px=200000.):
    """Sample at ~1 source pixel so long manually drawn lines carry weight."""
    if not .5<=step_px<=4.:
        raise ValueError("Invalid S97 source-coordinate sampling spacing")
    d=np.linalg.norm(np.diff(p,axis=0),axis=1)
    arclength=np.insert(np.cumsum(d),0,0.)
    length=float(arclength[-1])
    if length>max_length_px:
        raise ValueError("S97 reference stroke exceeds bounded length")
    if length<.5:
        raise ValueError("S97 reference stroke too short to evaluate")
    n=max(2,int(math.ceil(length/step_px))+1)
    tick=np.linspace(0.,length,n)
    return np.column_stack((
        np.interp(tick,arclength,p[:,0]),
        np.interp(tick,arclength,p[:,1]),
    )),length


def validate_reference_annotation(review:dict,pack:dict):
    """Return clean, source-pinned human strokes. Reject machine-guided input."""
    if not isinstance(pack,dict) or pack.get("schema")!=SCHEMA:
        raise ValueError("Unsupported S97 source-photo reference pack")
    if not isinstance(review,dict) or review.get("schema")!=SCHEMA:
        raise ValueError("Unsupported S97 human-reference schema")
    for key in ("source_bmp_name","source_bmp_sha256","image_shape_yx",
                "roi_xyxy_source","source_reference_pack_sha256"):
        expected=pack.get(key) if key!="source_reference_pack_sha256" else None
        if key=="source_reference_pack_sha256":
            continue  # Checked against immutable report file by the CLI.
        if review.get(key)!=expected:
            raise ValueError("S97 reference belongs to a different BMP/ROI")
    if (review.get("image_was_unmodified_original") is not True
        or review.get("machine_proposals_visible_during_annotation") is not False
        or review.get("full_exhaustive_negative_segmentation") is not False):
        raise ValueError("Human source review must be original-only and non-exhaustive")
    reviewer=review.get("reviewer_id")
    if (not isinstance(reviewer,str) or not 1<=len(reviewer)<=80
        or any(ord(c)<32 for c in reviewer)):
        raise ValueError("Need a nonempty bounded expert reviewer identifier")
    h,w=pack["image_shape_yx"]
    _positive_integer(h,"image height")
    _positive_integer(w,"image width")
    roi=pack.get("roi_xyxy_source")
    if (not isinstance(roi,list) or not roi
        or any(not isinstance(box,list) or len(box)!=4
               or any(type(z) is not int for z in box)
               or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h)
               for box in roi)):
        raise ValueError("Malformed S97 source ROIs")
    rows=review.get("strokes")
    if not isinstance(rows,list) or len(rows)>100000:
        raise ValueError("S97 stroke annotation list invalid")
    if (review.get("data_source")!="INDEPENDENT_OPERATOR_ORIGINAL_BMP_ONLY"
        or review.get("reference_semantics")!="PARTIAL_VISIBLE_2D_STROKES_NO_PARTICLE_GT"):
        raise ValueError("S97 source must declare unbiased partial 2D reference")
    ids=set()
    cleaned=[]
    for raw in rows:
        if not isinstance(raw,dict):
            raise ValueError("S97 stroke record is not an object")
        sid=raw.get("stroke_id")
        if (not isinstance(sid,str) or not 1<=len(sid)<=100 or sid in ids
            or any(ord(c)<32 for c in sid)):
            raise ValueError("Duplicate/invalid S97 expert stroke ID")
        ids.add(sid)
        k=raw.get("kind")
        certainty=raw.get("certainty")
        if k not in KINDS or certainty not in CERTAINTY:
            raise ValueError("Invalid S97 human type/certainty")
        instance=raw.get("visual_instance_id")
        if (not isinstance(instance,str) or not 1<=len(instance)<=96
            or any(ord(c)<32 for c in instance)):
            raise ValueError("S97 needs an explicit bounded visual instance ID")
        pts=_point_array(raw.get("points_xy_source_px"))
        if not (np.all((pts[:,0]>=0)&(pts[:,0]<w))
                and np.all((pts[:,1]>=0)&(pts[:,1]<h))):
            raise ValueError("S97 human stroke outside exact source BMP")
        if not any(np.all((pts[:,0]>=x0)&(pts[:,0]<x1)&
                              (pts[:,1]>=y0)&(pts[:,1]<y1))
                   for x0,y0,x1,y1 in roi):
            raise ValueError("S97 human stroke must belong to one declared reviewed ROI")
        dist=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        if np.max(dist)>32.:
            raise ValueError("S97 operator sampling >32px between clicks: risk of straightening a coil")
        length=float(dist.sum())
        if length<1.:
            raise ValueError("S97 human stroke length too short")
        if length>200000:
            raise ValueError("S97 unbounded source trace length")
        note=raw.get("note","")
        if not isinstance(note,str) or len(note)>600:
            raise ValueError("S97 note is invalid or too long")
        cleaned.append({
            "stroke_id":sid,
            "kind":k,
            "certainty":certainty,
            "visual_instance_id":instance,
            "points_xy_source_px":np.round(pts,4).tolist(),
            "approx_source_length_px":round(length,3),
            "note":note,
            "human_source":"ORIGINAL_PHOTO_2D_PARTIAL_REFERENCE_NOT_PHYSICS",
        })
    return cleaned


def _missed_runs(covered:np.ndarray,spacing:float,*,min_run_px:float=4.):
    missing=~covered
    runs=[]
    active=False
    begin=0
    for i,flag in enumerate(missing):
        if flag and not active:
            begin=i
            active=True
        elif not flag and active:
            runs.append((begin,i-1))
            active=False
    if active:
        runs.append((begin,len(missing)-1))
    # Arc samples are ~1px apart; count missed arclength conservatively.
    spans=[((b-a+1)*spacing,a,b) for a,b in runs]
    return {
        "max_uncovered_span_px":round(max([x[0] for x in spans],default=0.),3),
        "uncovered_spans_at_least_4px":sum(x[0]>=min_run_px for x in spans),
        "uncovered_span_source_sample_indices":[
            [int(a),int(b)] for length,a,b in spans if length>=min_run_px
        ][:100],
        "missed_span_list_truncated_to_100":sum(
            length>=min_run_px for length,_,_ in spans
        )>100,
    }


def _source_observed_points(components,max_points=2000000):
    source=[]
    n=0
    ids=set()
    for c in components:
        cid=c["component_id"]
        if cid in ids:raise ValueError("Duplicate S97 source component")
        ids.add(cid)
        xy=_point_array(c["points_xy"],min_points=2,max_points=1000000)
        n+=len(xy)
        if n>max_points:
            raise ValueError("S97 source centerline >2M samples: refuse silent truncation")
        source.append(xy)
    return np.concatenate(source) if source else np.empty((0,2),float)


def evaluate_partial_positive_coverage(
    reference_strokes:list[dict],
    observed_components:list[dict],
    *,
    radius_px:float=2.,
    step_px:float=1.,
):
    """Pure comparable observed-S86 metric on exactly the same human samples."""
    if not .5<=radius_px<=10.:
        raise ValueError("S97 tolerance radius out of bounds")
    pts=_source_observed_points(observed_components)
    tree=cKDTree(pts) if len(pts) else None
    per_stroke=[]
    denominator=numerator=0
    visual_total=0
    seen=set()
    for r in reference_strokes:
        ident=r["stroke_id"]
        if ident in seen:raise ValueError("Duplicated human reference stroke")
        seen.add(ident)
        if r["kind"] not in POSITIVE_KINDS or r["certainty"]!="CLEAR":
            continue
        reference=_point_array(r["points_xy_source_px"])
        sampled,length=_resample_polyline(reference,step_px=step_px)
        if tree is not None:
            dist=tree.query(sampled,k=1,workers=1)[0]
            coverage=np.asarray(dist<=radius_px+1e-9,dtype=bool)
        else:
            coverage=np.zeros(len(sampled),dtype=bool)
        count=int(coverage.sum())
        spacing=length/(len(sampled)-1)
        missing=_missed_runs(coverage,spacing)
        per_stroke.append({
            "stroke_id":ident,
            "kind":r["kind"],
            "human_visual_instance_id":r["visual_instance_id"],
            "reference_length_px":round(length,3),
            "reference_samples":len(sampled),
            "covered_reference_samples":count,
            "observed_S86_centerline_coverage":round(count/len(sampled),6),
            **missing,
        })
        denominator+=len(sampled)
        numerator+=count
        visual_total+=length
    return {
        "schema":EVALUATION_SCHEMA,
        "status":"PARTIAL_HUMAN_POSITIVE_ONLY_OBSERVED_CENTERLINE_EVALUATION",
        "reference_clear_track_or_coil_strokes":len(per_stroke),
        "human_positive_reference_length_px":round(visual_total,3),
        "reference_samples_total":denominator,
        "observed_centerline_covered_reference_samples":numerator,
        "observed_positive_centerline_coverage":(
            round(numerator/denominator,6) if denominator else None
        ),
        "tolerance_radius_px":radius_px,
        "reference_resampling_step_px":step_px,
        "source_observed_centerline_points":len(pts),
        "per_human_stroke":per_stroke,
        "inferred_bridges_excluded_from_observed_recall":True,
        "unknown_and_unreviewed_photo_pixels_not_negative_GT":True,
        "true_precision_fpr_f1_unavailable_without_exhaustive_negative_GT":True,
        "physical_particle_identity_and_3D_truth_not_established":True,
        "no_independent_film_level_confidence_intervals_from_one_photo":True,
    }


def paired_coverage_contrasts(runs:list[dict]):
    """Descriptive same-photo comparison, NOT bootstrap independent statistics."""
    if not runs:
        raise ValueError("S97 needs at least one candidate model")
    ids=[[x["stroke_id"] for x in run["per_human_stroke"]] for run in runs]
    if any(x!=ids[0] for x in ids[1:]):
        raise ValueError("Not the exact same expert reference stroke denominator")
    counts=[run["reference_samples_total"] for run in runs]
    if any(x!=counts[0] for x in counts[1:]):
        raise ValueError("S97 reference sample counts differ between model variants")
    paired=[]
    baseline=runs[0]
    for r in runs[1:]:
        aa=baseline["observed_positive_centerline_coverage"]
        bb=r["observed_positive_centerline_coverage"]
        paired.append({
            "baseline":baseline["candidate_name"],
            "challenger":r["candidate_name"],
            "baseline_coverage":aa,
            "challenger_coverage":bb,
            "descriptive_delta_pp":(
                round((bb-aa)*100.,4) if aa is not None and bb is not None else None
            ),
            "same_photo_same_human_reference":True,
            "no_statistical_significance_claim":True,
        })
    return paired
