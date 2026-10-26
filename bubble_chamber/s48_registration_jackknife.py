"""S48: position-dependent 2D registration sensitivity, not native physics truth.

The six leave-one-fiducial-out transforms are alternative hypotheses, NOT a
confidence interval or camera calibration. Both private photos are seen.
All outputs are inference-only advisories: never use them as heldout labels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .s39_machine_auto import ridge_evidence, sample_bilinear
from .s45_registered_unknown import repeat_advisory
from .s44_crossframe_persistence import fit_common_fiducials

SCHEMA = "s48_fiducial_jackknife_image_advisory_v1"
MIN_VARIANTS_FOR_INTERPRETATION = 4
STABILITY_VARIANTS = 4


def leave_one_fiducial_out(labels: dict) -> dict:
    """Use exactly the S44 2D similarity family. Never fit to test pixels."""
    full = fit_common_fiducials(labels)
    ids = tuple(f"f{i}" for i in range(1, 7))
    locations = {}
    for photo in ("005", "006"):
        name = f"H50Z_030_01_{photo}.bmp"
        if name not in labels:
            raise ValueError("Missing original fiducial photo")
        fids = {str(q["label"]): np.array([q["x_px"], q["y_px"]], dtype=float)
                for q in labels[name]["keypoints"] if str(q["label"]).startswith("f")}
        if not all(k in fids for k in ids):
            raise ValueError("Missing exactly matched common fiducials")
        locations[photo] = np.array([fids[k] for k in ids], dtype=np.float32)
    src, dst = locations["005"], locations["006"]
    full_m = full["forward_005_to_006_PRIVATE"]
    alternatives = []
    residuals = []
    for i, key in enumerate(ids):
        sx = np.delete(src, i, axis=0)
        sy = np.delete(dst, i, axis=0)
        model, inliers = cv2.estimateAffinePartial2D(
            sx, sy, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if model is None or inliers is None or not np.isfinite(model).all():
            raise ValueError("Insufficient fiducials for stable alternate registration")
        if int(inliers.sum()) < 4 or abs(float(np.linalg.det(model[:, :2]))) < 1e-7:
            raise ValueError("Unstable leave-one-out transform")
        pred = model[:, :2] @ src[i] + model[:, 2]
        err = float(np.linalg.norm(pred - dst[i]))
        alternatives.append(model)
        residuals.append(err)
    return {"full_forward_PRIVATE": full_m,
            "loo_forward_PRIVATE": alternatives,
            "heldout_identifiers": ids,
            "heldout_errors_px": residuals,
            "heldout_mean_px": float(np.mean(residuals)),
            "heldout_max_px": float(np.max(residuals)),
            "in_sample_mean_px": float(full["mean_common_residual_px"]),
            "in_sample_max_px": float(full["max_common_residual_px"]),
            "physical_3d_calibration_performed": False}


def photo_transforms(estimate: dict, photo_id: str) -> tuple[np.ndarray, list[np.ndarray]]:
    if photo_id not in ("005", "006"):
        raise ValueError("Only source photos 005 and 006")
    full = estimate["full_forward_PRIVATE"]
    models = estimate["loo_forward_PRIVATE"]
    if photo_id == "005":
        return full, models
    return cv2.invertAffineTransform(full), [cv2.invertAffineTransform(m) for m in models]


def shifted_path_for_registration(path_xy: np.ndarray, origin_xy: list,
                                  full: np.ndarray, alternate: np.ndarray) -> np.ndarray:
    """Return the same source-grid path predicted by alternate registration.

    If destination full transform maps X to Y, then Y_alt - Y_full is mapped
    back via the inverse linear part of the full model. No image pixels used.
    """
    p = np.asarray(path_xy, dtype=float)
    full = np.asarray(full, dtype=float)
    alternate = np.asarray(alternate, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all()
            or origin.shape != (2,) or not np.isfinite(origin).all()
            or full.shape != (2, 3) or alternate.shape != (2, 3)
            or not np.isfinite(full).all() or not np.isfinite(alternate).all()):
        raise ValueError("Invalid candidate registration provenance")
    linear = full[:, :2]
    if abs(float(np.linalg.det(linear))) < 1e-8:
        raise ValueError("Singular primary registration")
    global_xy = p + origin
    delta_dest = (global_xy @ (alternate[:, :2] - linear).T
                  + alternate[:, 2] - full[:, 2])
    delta_source = delta_dest @ np.linalg.inv(linear).T
    return p + delta_source


def ridge_binary(gray: np.ndarray) -> np.ndarray:
    """Same threshold/tolerance as S44 ridge_near_path, computed once per tile."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Native crop must be uint8 grayscale")
    score = ridge_evidence(gray)["score"]
    threshold = float(np.percentile(score, 95))
    if threshold <= 0:
        pos = score[score > 0]
        if pos.size == 0:
            return np.zeros(gray.shape, dtype=np.uint8)
        threshold = float(np.quantile(pos, .30))
    binary = ((score >= threshold) & (score > 0)).astype(np.uint8)
    return cv2.dilate(binary, np.ones((5, 5), dtype=np.uint8))


def ridge_path_support(binary: np.ndarray, points: np.ndarray) -> float | None:
    p = np.asarray(points, dtype=float)
    if binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("Invalid ridge support image")
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.isfinite(p).all():
        raise ValueError("Invalid candidate path")
    # No clipped-border sampling: an alternative is ineligible if part of the
    # curve leaves the valid image region. Do not cherry-pick a subset of it.
    if (np.any(p[:, 0] < 3) or np.any(p[:, 1] < 3)
            or np.any(p[:, 0] > binary.shape[1] - 4)
            or np.any(p[:, 1] > binary.shape[0] - 4)):
        return None
    return float(np.mean(sample_bilinear(binary, p) >= .5))


def assess_candidate(gray_src: np.ndarray, gray_other_registered: np.ndarray,
                     candidate: dict, s44_evidence: dict, origin_xy: list,
                     full: np.ndarray, variants: list[np.ndarray],
                     *, precomputed_destination_binary: np.ndarray | None = None) -> dict:
    if candidate.get("draft_id") != s44_evidence.get("draft_id"):
        raise ValueError("S39/S44 candidate identity mismatch")
    if (gray_src.ndim != 2 or gray_src.dtype != np.uint8
            or gray_other_registered.dtype != np.uint8
            or gray_src.shape != gray_other_registered.shape
            or len(variants) != 6):
        raise ValueError("Invalid registered source and six variant models")
    path = np.asarray(candidate.get("points_xy"), dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("Missing original candidate geometry")
    if not (np.all(path >= 0) and np.all(path[:, 0] < gray_src.shape[1])
            and np.all(path[:, 1] < gray_src.shape[0])):
        raise ValueError("Original geometry outside native crop")
    old = repeat_advisory(s44_evidence)
    dest_support = (precomputed_destination_binary if precomputed_destination_binary is not None
                    else ridge_binary(gray_other_registered))
    src_value = s44_evidence["source_multiscale_ridge_coverage"]
    control = s44_evidence["displaced_same_shape_counterpart_control_coverage"]
    fid = s44_evidence["known_fiducial_uncertainty_NOT_NATIVE_TRUTH"]
    supports = []
    drifts = []
    hits = []
    for alternate in variants:
        q = shifted_path_for_registration(path, origin_xy, full, alternate)
        drifts.append(float(np.max(np.linalg.norm(q - path, axis=1))))
        observed = ridge_path_support(dest_support, q)
        supports.append(observed)
        hits.append(bool(observed is not None and control is not None and not fid
                         and min(src_value, observed) >= .70
                         and min(src_value, observed) - control >= .20))
    available = sum(s is not None for s in supports)
    positive = sum(hits)
    baseline = bool(old["registered_repeat_risk"])
    if fid:
        status = "FIDUCIAL_GUARD_IS_SEPARATE"
    elif available < MIN_VARIANTS_FOR_INTERPRETATION:
        status = "INSUFFICIENT_VALID_JACKKNIFE_POSITIONS"
    elif baseline and positive >= STABILITY_VARIANTS:
        status = "BASELINE_REPEAT_STABLE_ACROSS_POSES"
    elif baseline:
        status = "BASELINE_REPEAT_REGISTRATION_SENSITIVE"
    elif positive:
        status = "POSE_ONLY_POSSIBLE_REPEAT_AMBIGUOUS"
    else:
        status = "NO_STRONG_IMAGE_REPEAT_EVIDENCE_NOT_PARTICLE_TRUTH"
    return {"draft_id": candidate["draft_id"], "registration_status": status,
            "S45_original_repeat_risk": baseline,
            "jackknife_variant_repeat_hits": int(positive),
            "jackknife_variant_evaluable": int(available),
            "jackknife_max_path_displacement_px_PRIVATE": max(drifts),
            "variant_support_PRIVATE": supports,
            "new_pose_only_warning": bool(not baseline and positive and not fid),
            "machine_warnings_only": True, "native_track_truth": "UNKNOWN",
            "source_and_target_photos_seen_transductive": True,
            "allowed_for_heldout_training": False,
            "physical_auto_join_authorized": False}


def annotate_graph(graph: dict, s40_graph: dict, candidate_details: dict[str, dict]) -> dict:
    nodes = graph.get("nodes")
    edges = graph.get("candidate_edges")
    s40_edges = s40_graph.get("candidate_edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(s40_edges, list):
        raise ValueError("Incomplete graph provenance")
    if len(edges) != len(s40_edges):
        raise ValueError("Fiducial and graph edge sequence differs")
    flags = []
    for n in nodes:
        original = n.get("source_candidate_ids")
        if not isinstance(original, list) or not original:
            raise ValueError("Missing graph node candidate provenance")
        if any(k not in candidate_details for k in original):
            raise ValueError("S48 graph node refers to missing candidate")
        flags.append(any(candidate_details[k]["S45_original_repeat_risk"]
                         or candidate_details[k]["new_pose_only_warning"] for k in original))
    counters = {"edge_count": 0, "fiducial_risk": 0, "S45_original_repeat_risk": 0,
                "S48_additional_pose_only_risk": 0, "union_any": 0,
                "unflagged_still_unverified": 0, "physics_auto_joins": 0}
    private = []
    for old, earlier in zip(edges, s40_edges):
        i, j = old.get("i"), old.get("j")
        if (not isinstance(i, int) or not isinstance(j, int) or i == j
                or not (0 <= i < len(nodes) and 0 <= j < len(nodes))
                or (i, j) != (earlier.get("i"), earlier.get("j"))
                or not isinstance(earlier.get("fiducial_overlap"), bool)):
            raise ValueError("Graph cannot be matched to S40 guard")
        fid = earlier["fiducial_overlap"]
        prior = any(any(candidate_details[k]["S45_original_repeat_risk"]
                        for k in nodes[z]["source_candidate_ids"]) for z in (i, j))
        new = bool(flags[i] or flags[j]) and not prior
        union = bool(fid or prior or new)
        counters["edge_count"] += 1
        counters["fiducial_risk"] += int(fid)
        counters["S45_original_repeat_risk"] += int(prior)
        counters["S48_additional_pose_only_risk"] += int(new and not fid)
        counters["union_any"] += int(union)
        counters["unflagged_still_unverified"] += int(not union)
        private.append({"i": i, "j": j, "fiducial": fid, "old_repeat": prior,
                        "pose_only": new, "union_advisory": union,
                        "physical_auto_join_authorized": False})
    return {"counters": counters, "per_edge_PRIVATE": private}
