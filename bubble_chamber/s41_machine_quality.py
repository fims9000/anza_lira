"""S41: photometric stability and fiducial-safe machine pseudo-labels.

The output is an *uncertainty annotation* of existing H50Z machine proposals.
Neither a high support score nor its robustness means a native particle track.
Original machine labels remain immutable; 255 is unknown, never negative.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .s39_machine_auto import ridge_evidence, fixed_budget_mask
from .s40_fiducial_safety import fiducial_guard

SCHEMA = "s41_unverified_machine_stability_and_unknown_v1"
UNKNOWN = np.uint8(255)


def exposure_views(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Fixed photometric challenges; not training augmentation or native GT."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 96:
        raise ValueError("Expected H50Z/synthetic uint8 grayscale image")
    x = gray.astype(np.float32) / 255.0
    return {
        "identity": gray.copy(),
        "gamma_0p78": np.rint(255.0 * np.power(x, 0.78)).astype(np.uint8),
        "gamma_1p30": np.rint(255.0 * np.power(x, 1.30)).astype(np.uint8),
        "blur_sigma_1p1": cv2.GaussianBlur(gray, (0, 0), 1.1),
    }


def ridge_support_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Same fixed activation budget per view, not image-dependent thresholds."""
    maps = {}
    for key, view in exposure_views(gray).items():
        score = ridge_evidence(view)["score"]
        binary = fixed_budget_mask(score, fraction=0.035)
        # tolerate <=2 pixel centerline displacement in recovered machine proposals
        maps[key] = (cv2.dilate(binary.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    return maps


def _validate_polyline(points: list | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    h, w = shape
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 2
            or not np.isfinite(p).all() or (p < 0).any()
            or (p[:, 0] >= w).any() or (p[:, 1] >= h).any()):
        raise ValueError("Out-of-frame or invalid machine centerline")
    return p


def candidate_stability(points: list | np.ndarray, support_maps: dict[str, np.ndarray]) -> dict[str, Any]:
    if set(support_maps) != {"identity", "gamma_0p78", "gamma_1p30", "blur_sigma_1p1"}:
        raise ValueError("Missing preregistered photometric challenges")
    shape = support_maps["identity"].shape
    if any(m.shape != shape or m.dtype != bool for m in support_maps.values()):
        raise ValueError("Inconsistent support map")
    p = _validate_polyline(points, shape)
    ix = np.clip(np.rint(p[:, 0]).astype(int), 0, shape[1] - 1)
    iy = np.clip(np.rint(p[:, 1]).astype(int), 0, shape[0] - 1)
    cover = {k: float(np.mean(v[iy, ix])) for k, v in support_maps.items()}
    baseline = cover["identity"]
    # Fixed heuristic, NOT learned or calibrated as an actual error probability.
    stable = baseline >= 0.35 and min(cover.values()) >= 0.25
    return {"pixel_ridge_support_fraction_by_view": {k: round(v, 4) for k, v in cover.items()},
            "all_views_min_support": round(min(cover.values()), 4),
            "photometric_machine_stable_NOT_GT": bool(stable),
            "native_particle_identity": "UNKNOWN"}


def uncertainty_region(shape: tuple[int, int], candidates: list[dict],
                       stability: list[dict], zones: list[dict],
                       origin_xy: tuple | list, photo: str) -> tuple[np.ndarray, list[dict]]:
    """Mask only physically known fiducial risk; photo instability is advisory.

    The scratch-origin negative control showed that stability is NOT a viable
    scratch-vs-track discriminator. Never use it to gate pseudo supervision.
    No previously unknown pixel is turned into a new positive or negative.
    """
    if len(candidates) != len(stability):
        raise ValueError("Candidate/stability mismatch")
    if len(shape) != 2 or min(shape) < 96:
        raise ValueError("Invalid target shape")
    if photo not in {"005", "006"}:
        raise ValueError("Unknown photo origin")
    origin = np.asarray(origin_xy, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("Invalid photo registration")
    protected = np.zeros(shape, dtype=np.uint8)
    by_candidate = []
    for candidate, support in zip(candidates, stability):
        path = _validate_polyline(candidate["points_xy"], shape)
        fid = fiducial_guard(path, origin, photo, zones)
        unstable = not support["photometric_machine_stable_NOT_GT"]
        if fid["fiducial_uncertainty_any"]:
            cv2.polylines(protected, [np.rint(path).astype(np.int32)], False, 1, 9)
        by_candidate.append({"draft_id": candidate["draft_id"],
                             "fiducial_risk": fid["fiducial_uncertainty_any"],
                             "unstable_photo_response": unstable,
                             "uncertain_not_assumed_negative": bool(fid["fiducial_uncertainty_any"]),
                             "photo_stability_is_advisory_only": True})
    for zone in zones:
        if zone["photo"] != photo:
            continue
        center = np.asarray(zone["center_xy_PRIVATE"], dtype=float) - origin
        if not np.isfinite(center).all():
            raise ValueError("Invalid fiducial position")
        radius = float(zone["machine_ambiguity_radius_px"])
        if not 0 < radius < 200:
            raise ValueError("Unsafe fiducial radius")
        c = tuple(np.rint(center).astype(int))
        cv2.circle(protected, c, int(np.ceil(radius)), 1, -1)
        arms = zone.get("arms_angles_rad_PRIVATE", [])
        if len(arms) == 2:
            span = float(zone.get("arm_half_span_px", 220))
            thickness = int(np.ceil(float(zone.get("arm_tolerance_px", 9)) * 2))
            for angle in arms:
                delta = span * np.array([np.cos(angle), np.sin(angle)])
                cv2.line(protected, tuple(np.rint(center-delta).astype(int)),
                         tuple(np.rint(center+delta).astype(int)), 1, max(1, thickness))
    return protected.astype(bool), by_candidate


def refine_pseudo_mask(original: np.ndarray, protected: np.ndarray) -> tuple[np.ndarray, dict]:
    """Monotone downgrade 0/1 -> 255; crucially *never invent* positive GT."""
    if (original.ndim != 2 or original.dtype != np.uint8
            or protected.shape != original.shape or protected.dtype != bool
            or not np.isin(original, [0, 1, 255]).all()):
        raise ValueError("Expected 0/1/255 pseudo mask and bool ambiguity map")
    refined = original.copy()
    before = {"positive": int(np.sum(original == 1)), "assumed_negative": int(np.sum(original == 0)),
              "unknown": int(np.sum(original == 255))}
    refined[protected] = UNKNOWN
    after = {"positive": int(np.sum(refined == 1)), "assumed_negative": int(np.sum(refined == 0)),
             "unknown": int(np.sum(refined == 255))}
    if after["positive"] > before["positive"] or after["assumed_negative"] > before["assumed_negative"]:
        raise AssertionError("S41 must only weaken supervision")
    return refined, {"before": before, "after": after, "protected_image_pixels": int(protected.sum()),
                     "positive_demoted_to_unknown": before["positive"] - after["positive"],
                     "negative_demoted_to_unknown": before["assumed_negative"] - after["assumed_negative"],
                     "native_expert_truth_created": 0}


def annotate_s41(gray: np.ndarray, old_mask: np.ndarray, candidates: list[dict],
                 zones: list[dict], origin_xy: tuple | list, photo: str) -> dict:
    if gray.shape != old_mask.shape:
        raise ValueError("Image/pseudo label dimensions mismatch")
    maps = ridge_support_maps(gray)
    scores = [candidate_stability(p["points_xy"], maps) for p in candidates]
    protect, flags = uncertainty_region(gray.shape, candidates, scores, zones, origin_xy, photo)
    refined, changes = refine_pseudo_mask(old_mask, protect)
    return {"schema": SCHEMA, "refined_mask_PRIVATE": refined,
            "protected_mask_PRIVATE": protect, "candidate_support_PRIVATE": scores,
            "candidate_flags_PRIVATE": flags, "changes": changes,
            "machine_line_count": len(candidates),
            "machine_stable_lines_UNVERIFIED": sum(x["photometric_machine_stable_NOT_GT"] for x in scores),
            "fiducial_risk_lines": sum(f["fiducial_risk"] for f in flags),
            "uncertain_lines": sum(f["uncertain_not_assumed_negative"] for f in flags),
            "native_physics_gt_count": 0, "physical_auto_join_allowed": False}
