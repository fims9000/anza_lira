"""Frozen, synthetic-only P2 segmentation comparison across optical stress regimes.

No physics-calibrated detector data are generated. Here 'clean' and 'clutter'
are photographic perturbation families, NOT claims about H2 or propane films.
Test seeds are disjoint from validation; no test threshold selection is allowed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from .pilot import PilotConfig, analyze_frame, extract_skeleton_segments


STUDY_VERSION = "bubble_p2_segmentation_synthetic_v1"
DOMAINS = ("clean_proxy", "clutter_proxy")
CASES = ("isolated", "gap", "crossing", "parallel", "artifact")


@dataclass(frozen=True)
class SegmentationProtocol:
    seed: int = 20261024
    size: int = 384
    val_scenes_per_domain: int = 12
    test_scenes_per_domain: int = 18
    contrast_thresholds: tuple[float, ...] = (12.0, 18.0, 24.0)
    ridge_fractions: tuple[float, ...] = (0.08, 0.12, 0.16)
    match_radius_px: int = 3
    min_component_area: int = 15
    min_segment_length_px: float = 28.0


def generate_scene(seed: int, domain: str, size: int = 384) -> dict:
    """Known visible 2D centerlines, variability in geometry, contrast and gaps."""
    if domain not in DOMAINS:
        raise ValueError(domain)
    if size < 256:
        raise ValueError("size must be >= 256")
    rng = np.random.default_rng(seed)
    case = CASES[seed % len(CASES)]
    clutter = domain == "clutter_proxy"
    std = rng.uniform(9, 15) if clutter else rng.uniform(3.5, 7.0)
    yy, xx = np.mgrid[:size, :size]
    photo = 219.0 + 10.0 * xx / size + 5.0 * yy / size + rng.normal(0.0, std, (size, size))
    observed_gt = np.zeros((size, size), np.uint8)
    complete_gt = np.zeros_like(observed_gt)
    main_y = rng.uniform(85, 215)
    main_slope = rng.uniform(-0.32, 0.32)
    main_curve = rng.uniform(-0.001, 0.001)
    curves = [(main_y, main_slope, main_curve)]
    if case in ("crossing", "parallel"):
        if case == "parallel":
            curves.append((main_y + rng.uniform(8, 18), main_slope + rng.uniform(-0.035, 0.035), main_curve))
        else:
            curves.append((main_y + rng.uniform(-18, 18), main_slope + rng.choice([-1, 1]) * rng.uniform(0.20, 0.58), rng.uniform(-0.0005, 0.0005)))
    for identity, (offset, slope, curve) in enumerate(curves):
        width = int(rng.integers(1, 4))
        darkness = rng.uniform(21, 48) if clutter else rng.uniform(50, 100)
        gap_center = rng.uniform(size * 0.42, size * 0.60)
        gap_width = rng.uniform(12, 30)
        previous = None
        for x in np.arange(24, size - 24, 1.0):
            y = offset + slope * (x - size / 2) + curve * (x - size / 2) ** 2
            if not (12 <= y < size - 12):
                previous = None
                continue
            pt = (int(round(x)), int(round(y)))
            if previous is not None:
                cv2.line(complete_gt, previous, pt, 255, 1)
                deleted = bool(identity == 0 and case == "gap" and abs(x - gap_center) <= gap_width / 2)
                missing = bool(rng.random() < (0.055 if clutter else 0.013))
                if not deleted and not missing:
                    cv2.line(observed_gt, previous, pt, 255, 1)
                    cv2.line(photo, previous, pt, 219 - darkness, width, cv2.LINE_8)
            previous = pt
    # Independent sensor/dust artifacts are deliberately NOT ground truth.
    speck_rate = rng.uniform(0.001, 0.004) if clutter else rng.uniform(0.0002, 0.0007)
    specks = rng.random((size, size)) < speck_rate
    photo[specks] -= rng.uniform(38, 95)
    if case == "artifact":
        for _ in range(3 if clutter else 1):
            x0 = int(rng.integers(0, size))
            x1 = int(np.clip(x0 + rng.integers(-45, 45), 0, size - 1))
            p0 = (x0, int(rng.integers(0, size // 5)))
            p1 = (x1, int(rng.integers(size // 2, size)))
            cv2.line(photo, p0, p1, float(rng.uniform(135, 185)), 1, cv2.LINE_8)
    if clutter:
        photo = cv2.GaussianBlur(photo.astype(np.float32), (0, 0), sigmaX=0.8)
    return {"image": np.uint8(np.clip(photo, 0, 255)), "observed_truth": observed_gt.astype(bool),
            "complete_truth": complete_gt.astype(bool), "domain": domain, "case": case,
            "scene_id": f"{domain}:{seed}", "seed": int(seed)}


def candidate_skeleton(mask: np.ndarray, min_length_px: float) -> tuple[np.ndarray, int, list[np.ndarray]]:
    paths = extract_skeleton_segments(skeletonize(np.asarray(mask, bool)), min_length_px=min_length_px)
    accepted = np.zeros(mask.shape, bool)
    for path in paths:
        pix = np.rint(path).astype(int)
        accepted[pix[:, 0], pix[:, 1]] = True
    return accepted, len(paths), paths


def _band(center: np.ndarray, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    return binary_dilation(center, structure=(xx * xx + yy * yy <= radius * radius))


def visible_track_metrics(
    mask: np.ndarray,
    truth: np.ndarray,
    *,
    radius_px: int,
    min_length_px: float,
) -> dict:
    """Compare *surviving extracted fragments*, not unconstrained raw mask.

    Relative metrics here are pixel-tolerant and are not stereometric precision.
    """
    accepted, number, paths = candidate_skeleton(mask, min_length_px)
    truth = np.asarray(truth, bool)
    truth_band = _band(truth, radius_px)
    pred_band = _band(accepted, radius_px)
    matched = int((accepted & truth_band).sum())
    precision = matched / max(int(accepted.sum()), 1)
    recall = int((truth & pred_band).sum()) / max(int(truth.sum()), 1)
    spurious = 0
    for path in paths:
        coord = np.rint(path).astype(int)
        if np.mean(truth_band[coord[:, 0], coord[:, 1]]) < 0.35:
            spurious += 1
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accepted_centerline_px": int(accepted.sum()),
        "visible_reference_px": int(truth.sum()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "surviving_fragments": number, "spurious_fragments": spurious,
    }


def config(method: str, parameter: float, protocol: SegmentationProtocol) -> PilotConfig:
    return PilotConfig(segmentation_method=method, threshold=parameter if method == "contrast" else 18.0,
                       ridge_response_fraction=parameter if method == "frangi" else 0.12,
                       min_component_area=protocol.min_component_area,
                       min_segment_length_px=protocol.min_segment_length_px)


def _probe(scenes: list[dict], cfg: PilotConfig, radius_px: int) -> dict:
    details = []
    for item in scenes:
        t0 = perf_counter()
        result, mask, _ = analyze_frame(item["image"], cfg)
        dt = perf_counter() - t0
        m = visible_track_metrics(mask > 0, item["observed_truth"], radius_px=radius_px,
                                  min_length_px=cfg.min_segment_length_px)
        details.append({"scene_id": item["scene_id"], "case": item["case"], "domain": item["domain"],
                        **m, "runtime_seconds": dt, "joins_for_review": result["proposed_joins_count"],
                        "automatic_merges": result["automatic_join_count"]})
    macro = {key: round(float(np.mean([d[key] for d in details])), 6) for key in
             ("precision", "recall", "f1", "surviving_fragments", "spurious_fragments",
              "runtime_seconds", "joins_for_review")}
    macro["scene_count"] = len(scenes)
    return {"macro": macro, "by_case": details}


def run_segmentation_study(protocol: SegmentationProtocol = SegmentationProtocol()) -> dict:
    """Choose both method operating points on VAL only, evaluate once on TEST."""
    if protocol.val_scenes_per_domain < 1 or protocol.test_scenes_per_domain < 1:
        raise ValueError("need nonempty validation and test")
    def scenes(split: str, domain: str):
        offset = 0 if split == "validation" else 1000000
        count = protocol.val_scenes_per_domain if split == "validation" else protocol.test_scenes_per_domain
        return [generate_scene(protocol.seed + offset + i, domain, size=protocol.size) for i in range(count)]

    val_scenes = [x for domain in DOMAINS for x in scenes("validation", domain)]
    test_scenes = {domain: scenes("test", domain) for domain in DOMAINS}
    val_ids = {x["scene_id"] for x in val_scenes}
    test_ids = {x["scene_id"] for batch in test_scenes.values() for x in batch}
    if not val_ids.isdisjoint(test_ids):
        raise AssertionError("scene leakage")

    result = {"status": "SYNTHETIC_SEGMENTATION_ONLY", "version": STUDY_VERSION,
              "protocol": asdict(protocol), "split_policy": "non-overlapping scene IDs; same data for both algorithms; validation-only selection",
              "studies": {}, "unlabeled_real_accuracy": None}
    for method, parameters in (("contrast", protocol.contrast_thresholds), ("frangi", protocol.ridge_fractions)):
        validation = []
        for param in parameters:
            cfg = config(method, param, protocol)
            scored = _probe(val_scenes, cfg, protocol.match_radius_px)
            validation.append({"parameter": param, "macro": scored["macro"]})
        # Frozen selection: highest validation macro-F1; tie-break fewer spurious fragments,
        # then prefer larger threshold. Test is never inspected during selection.
        chosen = max(validation, key=lambda x: (x["macro"]["f1"], -x["macro"]["spurious_fragments"], x["parameter"]))
        cfg = config(method, chosen["parameter"], protocol)
        tests = {domain: _probe(scenes, cfg, protocol.match_radius_px) for domain, scenes in test_scenes.items()}
        result["studies"][method] = {"validation_grid": validation, "selected_parameter": chosen["parameter"],
                                      "selected_configuration": asdict(cfg), "test": tests}
    # Post-hoc paired uncertainty description. It does not select parameters.
    # Statistical unit is the synthetic SCENE, not thousands of pixels.
    rng = np.random.default_rng(protocol.seed + 1)
    comparison = {}
    for domain in DOMAINS:
        old = result["studies"]["contrast"]["test"][domain]["by_case"]
        new = result["studies"]["frangi"]["test"][domain]["by_case"]
        if [r["scene_id"] for r in old] != [r["scene_id"] for r in new]:
            raise AssertionError("paired test scenes differ")
        differences = np.asarray([b["f1"] - a["f1"] for a, b in zip(old, new)], dtype=float)
        resamples = rng.integers(0, len(differences), size=(5000, len(differences)))
        boot = differences[resamples].mean(axis=1)
        comparison[domain] = {
            "frangi_minus_contrast_mean_scene_f1": round(float(differences.mean()), 6),
            "paired_scene_bootstrap_CI95": [round(float(x), 6) for x in np.quantile(boot, (0.025, 0.975))],
            "bootstrap_repetitions": 5000,
            "statistical_unit": "synthetic_scene", "scene_count": len(differences),
            "frangi_better_scene_count": int((differences > 0).sum()),
            "contrast_better_scene_count": int((differences < 0).sum()),
            "ties": int((differences == 0).sum()),
        }
    result["paired_test_comparison"] = comparison
    result["interpretation_warning"] = (
        "Test scores are synthetic photography-proxy only, with observed-trace GT. "
        "No true segmentation masks for real bubble chamber photos, no stereo, "
        "no per-track identity, no human-truth review, no H2/C3H8 transfer evidence. "
        "Any detector operating point requires recalibration on labeled films."
    )
    return result
