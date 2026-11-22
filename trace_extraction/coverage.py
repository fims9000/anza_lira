"""Coverage-oriented full-slice trace evidence, separate from link acceptance.

Segmentation masks describe *evidence*. Weak pixels are never promoted to
fault-instance identities or automatically connected across gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .graph import extract_trace_graph


@dataclass(frozen=True)
class CoverageLayers:
    high_confidence: np.ndarray
    supported_weak: np.ndarray
    review_weak: np.ndarray
    possible_trace: np.ndarray
    thresholds: tuple[float, float]


def _probability(probability):
    p=np.asarray(probability,dtype=np.float32)
    if p.ndim!=2 or not p.size or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Expected finite 2-D probability values in [0,1]")
    return p


def classify_trace_evidence(probability, *, low=.35, high=.75,
                            min_seed_pixels=2, connectivity=8,
                            max_support_distance_px=24.):
    """Keep weak evidence distinct, whether or not it touches a strong seed.

    Weak evidence attached to a high-confidence component is 'supported';
    isolated low-confidence structures are for review. Components are NOT
    geological track identities and connectivity in a probability mask is
    NOT permission to join physical faults.
    """
    p=_probability(probability)
    if (not 0 < low < high < 1 or not isinstance(min_seed_pixels,int)
            or min_seed_pixels<1 or connectivity not in (4,8)
            or not np.isfinite(max_support_distance_px)
            or max_support_distance_px<=0):
        raise ValueError("Invalid coverage thresholds or component policy")
    possible=p>=low
    high_mask=p>=high
    struct=ndi.generate_binary_structure(2,1 if connectivity==4 else 2)
    comp,n=ndi.label(possible,structure=struct)
    # A single massive dense background component is retained as uncertainty,
    # not silently promoted to a valid trace.
    counts=np.bincount(comp[high_mask],minlength=n+1)
    trusted=(counts>=min_seed_pixels)
    trusted[0]=False
    # Component membership alone can span a large noisy connected background.
    # Bound weak support by distance to high-confidence evidence; preserve
    # more distant candidate pixels in REVIEW instead of deleting them.
    if high_mask.any():
        distance=ndi.distance_transform_edt(~high_mask)
        near_high=distance<=max_support_distance_px
    else:
        near_high=np.zeros_like(high_mask,bool)
    supported=possible & trusted[comp] & near_high & ~high_mask
    review=possible & ~supported & ~high_mask
    return CoverageLayers(high_mask, supported,review,possible,(low,high))


def annotated_trace_coverage(probability, expert_mask, thresholds=(.85,.65,.45,.35),
                             *, tolerance_px=2):
    """Descriptive coverage of expert-marked centerline; no unlabeled negatives.

    Ground truth is read here AFTER prediction. The fraction of all predicted
    pixels near expert annotations is a *proximity indicator*, not precision,
    because unannotated genuine faults may exist.
    """
    p=_probability(probability)
    reference=np.asarray(expert_mask,dtype=bool)
    if reference.shape!=p.shape:
        raise ValueError("Reference shape mismatch")
    if (not thresholds or len(set(thresholds))!=len(thresholds)
            or any(not 0 < float(t) < 1 for t in thresholds)
            or tolerance_px<0):
        raise ValueError("Invalid coverage evaluation settings")
    centerline=skeletonize(reference)
    gt_count=int(centerline.sum())
    if not gt_count:
        return [{"threshold":float(t),"reference_centerline_pixels":0,
                 "annotated_centerline_coverage":None,
                 "reference_trace_fragment_count":0,
                 "annotated_fragment_recall_80pct":None,
                 "annotated_fragment_recall_95pct":None,
                 "foreground_fraction":float((p>=t).mean()),
                 "predicted_pixels_near_annotation_fraction":None}
                for t in thresholds]
    # Graph segments are annotated trace *fragments*, not physical fault IDs.
    # Do not merge disconnected components and call them geological truth.
    gt_graph=extract_trace_graph(centerline,border_margin=0)
    gt_fragments=[
        np.asarray(segment.pixels,dtype=np.int32)
        for segment in gt_graph.segments if segment.pixel_length>=8.
    ]
    dist_from_gt=ndi.distance_transform_edt(~centerline)
    results=[]
    for threshold in thresholds:
        mask=p>=threshold
        frac=float(mask.mean())
        # A reference point counts only if predicted evidence is within tolerance.
        if mask.any():
            dist=ndi.distance_transform_edt(~mask)
            recovery=float(np.mean(dist[centerline]<=tolerance_px))
            proximity=float(np.mean(dist_from_gt[mask]<=tolerance_px))
        else:
            recovery=0.
            proximity=None
            dist=None
        # A gap-free mask is not the same as high fragment/track recall.
        def fragment_recall(min_coverage):
            if not gt_fragments:
                return None
            if dist is None:
                return 0.
            passed=0
            for points in gt_fragments:
                ratio=float(np.mean(
                    dist[points[:,0],points[:,1]]<=tolerance_px))
                passed+=ratio>=min_coverage
            return passed/len(gt_fragments)
        results.append({
            "threshold":float(threshold),
            "reference_centerline_pixels":gt_count,
            "annotated_centerline_coverage":recovery,
            "reference_trace_fragment_count":len(gt_fragments),
            "annotated_fragment_recall_80pct":fragment_recall(.80),
            "annotated_fragment_recall_95pct":fragment_recall(.95),
            "foreground_fraction":frac,
            "predicted_pixels_near_annotation_fraction":proximity,
        })
    return results


def mode_layers(probability, mode="coverage"):
    """Operator-selected operating modes. Not fitted using expert labels."""
    config={
        "coverage":(.25,.70),
        "balanced":(.40,.80),
        "strict":(.60,.90),
    }
    if mode not in config:
        raise ValueError("Unknown operating mode")
    low,high=config[mode]
    return classify_trace_evidence(probability,low=low,high=high)
