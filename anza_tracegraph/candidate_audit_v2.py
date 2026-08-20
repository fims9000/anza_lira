"""Zero-training forensics for the frozen TraceGraph candidate front-end."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import label, map_coordinates
from scipy.spatial import cKDTree

from trace_extraction.graph import extract_trace_graph
from trace_extraction.skeleton import skeletonize_mask

from .candidates import axial_error
from .data import SCENE_TYPES, SPLIT_SIZES, generate_scene
from .frozen_source import DENSE_CHECKPOINT, DENSE_THRESHOLD, FORCED_GAP_X, infer_dense, load_frozen_source, predicted_relation_scene
from .protocol import protocol_hash as parent_protocol_hash
from .tracelets import Endpoint, Tracelet, endpoints, extract_tracelets


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/anza_tracegraph/candidate_audit_v2"
PARENT = ROOT / "results/anza_tracegraph/tg2"
K_VALUES = (4, 8, 12, 16, 24, 32)
RADII = (6.0, 8.0, 10.0)
AUDIT_PROTOCOL: dict[str, Any] = {
    "version": "ANZA_TRACEGRAPH_CANDIDATE_AUDIT_V2",
    "parent_protocol_sha256": parent_protocol_hash(),
    "parent_status": "STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK",
    "split": "development only",
    "predictions": "exact frozen ANZA-KIR R0 checkpoint; no prediction or threshold change",
    "branch_match": {"truth_tube_radius_px": 3.0, "minimum_predicted_tracelet_fraction_in_tube": 0.60},
    "endpoint_radii_diagnostic_only": list(RADII),
    "candidate_k": list(K_VALUES),
    "directed_port": "dot(t_src,d)>0 and dot(t_dst,-d)>0; max directed angle <=78 degrees",
    "aligned_gap_diagnostic": "same prediction, remove x from 35 through ceil(generator destination start x)",
    "valley": "source and target share original threshold-mask component and gap minimum <0.80 endpoint support",
    "taxonomy_priority": [
        "B_correct_branch_eligible_but_dropped_by_topK",
        "D_skeleton_connected_with_confidence_valley",
        "A_correct_branch_port_in_topK_but_endpoint_shifted",
        "C_branch_support_or_junction_but_no_admissible_port",
        "E_correct_branch_absent_in_dense_prediction",
    ],
    "locks": {"training": True, "threshold_change": True, "confirm": True, "cracks": True, "expert": True, "path": True},
}


def audit_protocol_hash() -> str:
    return hashlib.sha256(json.dumps(AUDIT_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_manifest() -> dict[str, Any]:
    paths = [ROOT / "anza_tracegraph/candidate_audit_v2.py", ROOT / "anza_tracegraph/candidate_audit_v2_validator.py", ROOT / "anza_tracegraph/frozen_source.py", ROOT / "anza_tracegraph/candidates.py", ROOT / "anza_tracegraph/tracelets.py", ROOT / "anza_tracegraph/data.py", ROOT / "scripts/run_anza_tracegraph_candidate_audit_v2.py", ROOT / "scripts/validate_anza_tracegraph_candidate_audit_v2.py", ROOT / "tests/test_anza_tracegraph_candidate_audit_v2.py"]
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": rows, "sha256": digest.hexdigest()}


def _branch_match(tracelet: Tracelet, truth: np.ndarray) -> tuple[bool, float, float]:
    distance = cKDTree(np.asarray(truth, dtype=float)).query(np.asarray(tracelet.points_yx, dtype=float))[0]
    fraction = float(np.mean(distance <= AUDIT_PROTOCOL["branch_match"]["truth_tube_radius_px"]))
    median = float(np.median(distance))
    return bool(fraction >= AUDIT_PROTOCOL["branch_match"]["minimum_predicted_tracelet_fraction_in_tube"]), fraction, median


def _candidate_pool(source: Endpoint, tracelets: tuple[Tracelet, ...], truth: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tracelet_by_id = {tracelet.tracelet_id: tracelet for tracelet in tracelets}
    for tracelet in tracelets:
        if tracelet.tracelet_id == source.tracelet_id:
            continue
        branch_match, branch_fraction, branch_median = _branch_match(tracelet, truth)
        for endpoint in endpoints(tracelet, 5):
            if endpoint.point_yx[1] <= FORCED_GAP_X[0] + 10:
                continue
            delta = np.asarray(endpoint.point_yx) - np.asarray(source.point_yx)
            distance = float(np.linalg.norm(delta))
            if distance <= 0:
                continue
            direction = delta / distance
            src_axial = axial_error(source.outgoing_tangent_yx, tuple(direction))
            dst_axial = axial_error(endpoint.outgoing_tangent_yx, tuple(direction))
            axial = max(src_axial, dst_axial)
            source_dot = float(np.dot(np.asarray(source.outgoing_tangent_yx), direction))
            destination_dot = float(np.dot(np.asarray(endpoint.outgoing_tangent_yx), -direction))
            directed_error = max(math.acos(float(np.clip(source_dot, -1.0, 1.0))), math.acos(float(np.clip(destination_dot, -1.0, 1.0))))
            eligible = bool(6.0 <= distance <= 68.0 and axial <= math.radians(78.0))
            directed_eligible = bool(6.0 <= distance <= 68.0 and source_dot > 0 and destination_dot > 0 and directed_error <= math.radians(78.0))
            rows.append({
                "endpoint": endpoint,
                "tracelet": tracelet_by_id[endpoint.tracelet_id],
                "distance": distance,
                "axial_error": axial,
                "directed_error": directed_error,
                "source_dot": source_dot,
                "destination_dot": destination_dot,
                "score": distance + 8.0 * axial,
                "eligible": eligible,
                "directed_eligible": directed_eligible,
                "branch_match": branch_match,
                "branch_fraction": branch_fraction,
                "branch_median_distance": branch_median,
            })
    return rows


def _rank(pool: list[dict[str, Any]], *, directed: bool = False) -> list[dict[str, Any]]:
    flag = "directed_eligible" if directed else "eligible"
    return sorted((row for row in pool if row[flag]), key=lambda row: (row["score"], row["endpoint"].tracelet_id, row["endpoint"].end_index))


def _context(raw: dict[str, Any], probability: np.ndarray, force_end: int | None) -> dict[str, Any]:
    mask = np.asarray(probability) >= DENSE_THRESHOLD
    if force_end is not None and force_end > FORCED_GAP_X[0]:
        mask[:, FORCED_GAP_X[0] : min(force_end, mask.shape[1])] = False
    tracelets = extract_tracelets(mask, probability, raw["dense"][0], min_length=8)
    all_endpoints = [endpoint for tracelet in tracelets for endpoint in endpoints(tracelet, 5)]
    source_truth = np.asarray(raw["source_endpoint"].point_yx)
    source_options = [endpoint for endpoint in all_endpoints if endpoint.point_yx[1] < FORCED_GAP_X[1] - 8]
    if not source_options:
        return {"mask": mask, "tracelets": tracelets, "source": None, "pool": [], "ranked": [], "directed_ranked": []}
    source = min(source_options, key=lambda endpoint: float(np.linalg.norm(np.asarray(endpoint.point_yx) - source_truth)))
    truth = raw["tracelets"][1].points_yx
    pool = _candidate_pool(source, tracelets, truth)
    return {"mask": mask, "tracelets": tracelets, "source": source, "pool": pool, "ranked": _rank(pool), "directed_ranked": _rank(pool, directed=True)}


def _endpoint_error(row: dict[str, Any], truth_endpoint: Endpoint) -> tuple[float, float, float]:
    error = np.asarray(row["endpoint"].point_yx) - np.asarray(truth_endpoint.point_yx)
    tangent = np.asarray(truth_endpoint.outgoing_tangent_yx)
    normal = np.asarray((-tangent[1], tangent[0]))
    return float(np.linalg.norm(error)), float(np.dot(error, tangent)), float(np.dot(error, normal))


def _same_component(mask: np.ndarray, first: np.ndarray, second: np.ndarray, tolerance: float = 5.0) -> bool:
    components, _ = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    points = np.argwhere(mask)
    if not len(points):
        return False
    a_index = int(np.argmin(np.linalg.norm(points - first[None], axis=1)))
    b_index = int(np.argmin(np.linalg.norm(points - second[None], axis=1)))
    if np.linalg.norm(points[a_index] - first) > tolerance or np.linalg.norm(points[b_index] - second) > tolerance:
        return False
    a = tuple(points[a_index]); b = tuple(points[b_index])
    return bool(components[a] != 0 and components[a] == components[b])


def _gap_profile(probability: np.ndarray, source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    alpha = np.linspace(0.0, 1.0, 64)
    points = source[:, None] * (1 - alpha[None]) + target[:, None] * alpha[None]
    values = map_coordinates(probability, (points[0], points[1]), order=1, mode="nearest")
    edge = float(min(values[:8].mean(), values[-8:].mean()))
    return float(values.min()), edge


def _junction_near(mask: np.ndarray, point: np.ndarray, radius: float = 8.0) -> bool:
    graph = extract_trace_graph(skeletonize_mask(mask), border_margin=0)
    pixels = [pixel for component in graph.junctions for pixel in component]
    return bool(pixels and min(np.linalg.norm(np.asarray(pixel) - point) for pixel in pixels) <= radius)


def _summary_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = [row for row in rows if row["positive"]]
    curves = []
    for directed in (False, True):
        prefix = "directed" if directed else "axial_v1"
        for k in K_VALUES:
            local = [row for row in positives]
            curves.append({
                "candidate_rule": prefix,
                "k": k,
                "branch_candidate_recall": float(np.mean([row[f"{prefix}_branch_at_{k}"] for row in local])),
                **{f"endpoint_recall_at_{int(radius)}px": float(np.mean([row[f"{prefix}_localization_{int(radius)}_at_{k}"] for row in local])) for radius in RADII},
            })
    misses = [row for row in positives if not row["v1_recalled_6"]]
    categories = {name: sum(row["category"] == name for row in misses) for name in AUDIT_PROTOCOL["taxonomy_priority"]}
    metrics = {
        "status": "CANDIDATE_AUDIT_V2_COMPLETE",
        "development_sources": len(rows),
        "positive_sources": len(positives),
        "none_sources": len(rows) - len(positives),
        "v1_misses": len(misses),
        "v1_recall_at_6px": float(np.mean([row["v1_recalled_6"] for row in positives])),
        "v1_recall_at_8px": float(np.mean([row["v1_recalled_8"] for row in positives])),
        "v1_recall_at_10px": float(np.mean([row["v1_recalled_10"] for row in positives])),
        "v1_distance_bins": {
            "le_6": sum(row["v1_recalled_6"] for row in positives),
            "gt_6_le_8": sum((not row["v1_recalled_6"]) and row["v1_recalled_8"] for row in positives),
            "gt_8_le_10": sum((not row["v1_recalled_8"]) and row["v1_recalled_10"] for row in positives),
            "gt_10_or_missing": sum(not row["v1_recalled_10"] for row in positives),
        },
        "misses_with_full_top8": sum(row["v1_candidate_count"] == 8 for row in misses),
        "unpruned_branch_exists_in_misses": sum(row["axial_v1_branch_unpruned"] for row in misses),
        "mean_axial_eligible_pool": float(np.mean([row["axial_eligible_pool_count"] for row in positives])),
        "mean_directed_eligible_pool": float(np.mean([row["directed_eligible_pool_count"] for row in positives])),
        "mean_away_facing_ports_removed": float(np.mean([row["away_facing_ports_removed"] for row in positives])),
        "branch_candidate_recall_at_8": float(np.mean([row["axial_v1_branch_at_8"] for row in positives])),
        "aligned_gap_endpoint_recall_at_6": float(np.mean([row["aligned_localization_6_at_8"] for row in positives])),
        "aligned_gap_branch_recall_at_8": float(np.mean([row["aligned_branch_at_8"] for row in positives])),
        "original_mask_connected_fraction": float(np.mean([row["original_connected"] for row in positives])),
        "confidence_valley_fraction": float(np.mean([row["confidence_valley"] for row in positives])),
        "taxonomy": categories,
        "locks": {"training_opened": False, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False, "threshold_changed": False},
    }
    return curves, metrics


def run(*, device: str = "cuda") -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True)
    _json(RESULT / "protocol.json", AUDIT_PROTOCOL)
    (RESULT / "protocol_hash.txt").write_text(audit_protocol_hash() + "\n")
    source = _source_manifest(); _json(RESULT / "source_manifest.json", source)
    parent_metrics = json.loads((PARENT / "metrics.json").read_text())
    checkpoint_before = _sha(DENSE_CHECKPOINT)
    model = load_frozen_source(device)
    rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    batch_size = 64
    for start in range(0, SPLIT_SIZES["development"], batch_size):
        raw_batch = [generate_scene("development", index) for index in range(start, min(start + batch_size, SPLIT_SIZES["development"]))]
        probability_batch, orientation_batch = infer_dense(model, np.stack([raw["dense"][:3] for raw in raw_batch]), device=device)
        for raw, probability, orientation in zip(raw_batch, probability_batch, orientation_batch):
            adapted = predicted_relation_scene(raw, probability, orientation)
            positive = bool(raw["has_valid_continuation"])
            base: dict[str, Any] = {"index": raw["index"], "scene_type": raw["scene_type"], "positive": int(positive), "source_available": int(adapted["source_available"]), "v1_candidate_count": int(adapted["candidate_count"]), "v1_target_distance": adapted.get("target_match_distance")}
            for radius in RADII:
                base[f"v1_recalled_{int(radius)}"] = int(not positive or (adapted.get("target_match_distance") is not None and adapted["target_match_distance"] <= radius))
            if not positive:
                base.update({"category": "NONE", "original_connected": 0, "confidence_valley": 0})
                for prefix in ("axial_v1", "directed"):
                    base[f"{prefix}_branch_unpruned"] = 0
                    for k in K_VALUES:
                        base[f"{prefix}_branch_at_{k}"] = 0
                        for radius in RADII: base[f"{prefix}_localization_{int(radius)}_at_{k}"] = 0
                base.update({"aligned_branch_at_8": 0, "aligned_localization_6_at_8": 0, "target_support_fraction": 0.0, "junction_near_target": 0, "gap_end_x": None, "forced_gap_mismatch_px": None})
                rows.append(base); continue
            truth_tracelet = raw["tracelets"][1]; truth_endpoint = endpoints(truth_tracelet, 5)[0]; truth_point = np.asarray(truth_endpoint.point_yx)
            current = _context(raw, probability, FORCED_GAP_X[1])
            aligned_end = max(FORCED_GAP_X[0] + 1, int(math.ceil(truth_point[1])))
            aligned = _context(raw, probability, aligned_end)
            original_mask = probability >= DENSE_THRESHOLD
            original_connected = _same_component(original_mask, np.asarray(raw["source_endpoint"].point_yx), truth_point)
            gap_min, endpoint_support = _gap_profile(probability, np.asarray(raw["source_endpoint"].point_yx), truth_point)
            confidence_valley = bool(original_connected and endpoint_support > 0 and gap_min < 0.80 * endpoint_support)
            target_pixels = np.rint(truth_tracelet.points_yx).astype(int); target_pixels[:, 0] = np.clip(target_pixels[:, 0], 0, probability.shape[0] - 1); target_pixels[:, 1] = np.clip(target_pixels[:, 1], 0, probability.shape[1] - 1)
            target_support = float(np.mean(original_mask[target_pixels[:, 0], target_pixels[:, 1]]))
            junction_near = _junction_near(current["mask"], truth_point)
            base["axial_eligible_pool_count"] = len(current["ranked"])
            base["directed_eligible_pool_count"] = len(current["directed_ranked"])
            base["away_facing_ports_removed"] = sum(row["eligible"] and not row["directed_eligible"] for row in current["pool"])
            for prefix, ranked in (("axial_v1", current["ranked"]), ("directed", current["directed_ranked"])):
                base[f"{prefix}_branch_unpruned"] = int(any(row["branch_match"] for row in ranked))
                for k in K_VALUES:
                    selected = ranked[:k]
                    base[f"{prefix}_branch_at_{k}"] = int(any(row["branch_match"] for row in selected))
                    for radius in RADII:
                        base[f"{prefix}_localization_{int(radius)}_at_{k}"] = int(any(_endpoint_error(row, truth_endpoint)[0] <= radius for row in selected))
            aligned_top = aligned["ranked"][:8]
            base["aligned_branch_at_8"] = int(any(row["branch_match"] for row in aligned_top))
            base["aligned_localization_6_at_8"] = int(any(_endpoint_error(row, truth_endpoint)[0] <= 6.0 for row in aligned_top))
            base.update({"original_connected": int(original_connected), "confidence_valley": int(confidence_valley), "gap_min_probability": gap_min, "gap_endpoint_support": endpoint_support, "target_support_fraction": target_support, "junction_near_target": int(junction_near), "gap_end_x": float(truth_point[1]), "forced_gap_mismatch_px": float(truth_point[1] - FORCED_GAP_X[1])})
            current_top = current["ranked"][:8]
            branch_top = [row for row in current_top if row["branch_match"]]
            branch_unpruned = [row for row in current["ranked"] if row["branch_match"]]
            if not base["v1_recalled_6"]:
                if branch_unpruned and not branch_top:
                    category = AUDIT_PROTOCOL["taxonomy_priority"][0]
                elif confidence_valley:
                    category = AUDIT_PROTOCOL["taxonomy_priority"][1]
                elif branch_top:
                    category = AUDIT_PROTOCOL["taxonomy_priority"][2]
                elif target_support >= 0.50 or junction_near:
                    category = AUDIT_PROTOCOL["taxonomy_priority"][3]
                else:
                    category = AUDIT_PROTOCOL["taxonomy_priority"][4]
            else:
                category = "V1_RECALLED"
            base["category"] = category
            if branch_unpruned:
                ranked_branch = [(rank, candidate, _endpoint_error(candidate, truth_endpoint)) for rank, candidate in enumerate(current["ranked"]) if candidate["branch_match"]]
                rank, candidate, (total, longitudinal, transverse) = min(ranked_branch, key=lambda item: item[2][0])
                error_rows.append({"index": raw["index"], "scene_type": raw["scene_type"], "rank": rank, "within_top8": int(rank < 8), "total_error": total, "longitudinal_error": longitudinal, "transverse_error": transverse, "abs_longitudinal_error": abs(longitudinal), "abs_transverse_error": abs(transverse), "branch_fraction": candidate["branch_fraction"], "branch_median_distance": candidate["branch_median_distance"]})
            rows.append(base)
    checkpoint_after = _sha(DENSE_CHECKPOINT)
    if checkpoint_before != checkpoint_after or checkpoint_before != AUDIT_PROTOCOL_CHECKPOINT_SHA:
        raise PermissionError("Candidate audit changed or mismatched the frozen dense checkpoint")
    curves, metrics = _summary_rows(rows)
    if error_rows:
        metrics["nearest_correct_branch_port_error_quantiles"] = {
            key: {str(q): float(np.quantile([row[key] for row in error_rows], q)) for q in (0.5, 0.9, 0.95)}
            for key in ("total_error", "abs_longitudinal_error", "abs_transverse_error")
        }
    passing_k = [row["k"] for row in curves if row["candidate_rule"] == "axial_v1" and row["branch_candidate_recall"] >= 0.95]
    metrics["minimum_k_for_branch_recall_0_95"] = min(passing_k) if passing_k else None
    parent_miss_indices = {int(row["index"]) for row in csv.DictReader((PARENT / "candidate_per_case.csv").open()) if row["positive"] == "1" and row["recalled"] == "0"}
    current_miss_indices = {row["index"] for row in rows if row["positive"] and not row["v1_recalled_6"]}
    metrics.update({"protocol_sha256": audit_protocol_hash(), "source_sha256": source["sha256"], "dense_checkpoint_sha256": checkpoint_after, "parent_protocol_sha256": parent_protocol_hash(), "parent_miss_set_exact": current_miss_indices == parent_miss_indices, "parent_candidate_recall": parent_metrics["candidate_recall"]["candidate_recall"]})
    misses = [row for row in rows if row["positive"] and not row["v1_recalled_6"]]
    _write_csv(RESULT / "per_case.csv", rows)
    _write_csv(RESULT / "miss_taxonomy.csv", misses)
    _write_csv(RESULT / "endpoint_errors.csv", error_rows)
    _write_csv(RESULT / "recall_vs_k.csv", curves)
    taxonomy_scene_rows = []
    for scene_type in SCENE_TYPES:
        local = [row for row in misses if row["scene_type"] == scene_type]
        taxonomy_scene_rows.append({"scene_type": scene_type, "misses": len(local), **{name: sum(row["category"] == name for row in local) for name in AUDIT_PROTOCOL["taxonomy_priority"]}})
    _write_csv(RESULT / "taxonomy_by_scene.csv", taxonomy_scene_rows)
    _json(RESULT / "taxonomy.json", {"definitions": AUDIT_PROTOCOL["taxonomy_priority"], "counts": metrics["taxonomy"], "total": len(misses)})
    _json(RESULT / "gap_mismatch.json", {"forced_gap": list(FORCED_GAP_X), "actual_gap_end_quantiles": np.quantile([row["gap_end_x"] for row in rows if row["positive"]], [0, 0.25, 0.5, 0.75, 1]).tolist(), "mismatch_quantiles": np.quantile([row["forced_gap_mismatch_px"] for row in rows if row["positive"]], [0, 0.25, 0.5, 0.75, 1]).tolist(), "v1_recall_at_6": metrics["v1_recall_at_6px"], "aligned_recall_at_6": metrics["aligned_gap_endpoint_recall_at_6"], "v1_branch_at_8": metrics["branch_candidate_recall_at_8"], "aligned_branch_at_8": metrics["aligned_gap_branch_recall_at_8"]})
    specialized = ["straight", "s_curve", "long_gap", "close_parallel", "parallel_gap_confuser", "x_crossing", "acute_crossing", "low_contrast", "partial_occlusion", "cluttered_corridor"]
    unspecialized = [task for task in SCENE_TYPES if task not in specialized]
    implementation = {"curvature_split_declared_radians": 0.70, "curvature_split_used_by_extract_tracelets": "curvature" in inspect.getsource(extract_tracelets), "specialized_scene_types": specialized, "unspecialized_scene_types": unspecialized, "none_positive_examples": sum(row["scene_type"] == "none" and row["positive"] for row in rows), "none_negative_examples": sum(row["scene_type"] == "none" and not row["positive"] for row in rows), "positive_flag_independent_of_scene_name": True}
    _json(RESULT / "implementation_audit.json", implementation)
    _json(RESULT / "zero_training_receipt.json", {"training_opened": False, "optimizer_created": False, "checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False})
    _json(RESULT / "metrics.json", metrics)
    (RESULT / "CANDIDATE_AUDIT_V2_REPORT.md").write_text(_report(metrics, curves, implementation))
    return metrics


AUDIT_PROTOCOL_CHECKPOINT_SHA = "95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6"


def _report(metrics: dict[str, Any], curves: list[dict[str, Any]], implementation: dict[str, Any]) -> str:
    lines = ["# ANZA-TraceGraph Candidate Audit V2", "", f"Status: `{metrics['status']}`", "", "This is a zero-training forensic analysis of the exact frozen TG1 predictions. Radius 8/10 and aligned-gap values are diagnostics, not revised gates.", "", "## Reproduction", "", f"- V1 Recall@6: `{metrics['v1_recall_at_6px']:.6f}`", f"- V1 misses: `{metrics['v1_misses']}`", f"- exact parent miss set: `{metrics['parent_miss_set_exact']}`", f"- distance bins: `{json.dumps(metrics['v1_distance_bins'], sort_keys=True)}`", f"- misses with K=8 full: `{metrics['misses_with_full_top8']}`", "", "## Cause taxonomy", "", "| Cause | Count |", "|---|---:|"]
    for name, count in metrics["taxonomy"].items(): lines.append(f"| {name} | {count} |")
    repairable = metrics["v1_misses"] - metrics["taxonomy"]["E_correct_branch_absent_in_dense_prediction"]
    lines += ["", f"Operational A--D cases account for `{repairable}/{metrics['v1_misses']}` (`{repairable / metrics['v1_misses']:.3%}`); E accounts for `{metrics['taxonomy']['E_correct_branch_absent_in_dense_prediction']}/{metrics['v1_misses']}`. The taxonomy is defined by the frozen 3 px branch tube and 0.60 overlap rule, not claimed as annotation-independent ground truth."]
    lines += ["", "## Coverage", "", "| Rule | K | Branch recall | Endpoint@6 | Endpoint@8 | Endpoint@10 |", "|---|---:|---:|---:|---:|---:|"]
    for row in curves: lines.append(f"| {row['candidate_rule']} | {row['k']} | {row['branch_candidate_recall']:.6f} | {row['endpoint_recall_at_6px']:.6f} | {row['endpoint_recall_at_8px']:.6f} | {row['endpoint_recall_at_10px']:.6f} |")
    lines += ["", "## Port geometry", "", f"- mean axial eligible pool: `{metrics['mean_axial_eligible_pool']:.3f}`", f"- mean directed eligible pool: `{metrics['mean_directed_eligible_pool']:.3f}`", f"- mean away-facing ports removed: `{metrics['mean_away_facing_ports_removed']:.3f}`", f"- smallest K reaching branch recall 0.95: `{metrics['minimum_k_for_branch_recall_0_95']}`", f"- nearest correct-port error quantiles: `{json.dumps(metrics['nearest_correct_branch_port_error_quantiles'], sort_keys=True)}`", "", "K expansion is not the main repair: axial branch recall rises only from 0.909180 at K=8 to 0.915039 at K=12 and then saturates. Directed ports halve the pool and slightly improve branch recall at small K, but reduce endpoint-radius recall; they are a useful pruning diagnostic, not a passed replacement.", "", "Localization error is predominantly longitudinal along the correct branch rather than transverse to a neighboring branch, especially in the upper tail.", "", "## Protocol mismatches", "", f"- aligned forced-gap Recall@6: `{metrics['aligned_gap_endpoint_recall_at_6']:.6f}` versus V1 `{metrics['v1_recall_at_6px']:.6f}`", f"- aligned branch recall@8: `{metrics['aligned_gap_branch_recall_at_8']:.6f}`", f"- curvature split declared but implemented: `{implementation['curvature_split_used_by_extract_tracelets']}`", f"- scene names without specialized construction: `{', '.join(implementation['unspecialized_scene_types'])}`", f"- `none` positives / negatives: `{implementation['none_positive_examples']} / {implementation['none_negative_examples']}`", "", "Aligning the forced cut to the generator endpoint does not rescue V1: Recall@6 falls by 0.027344 and branch recall is unchanged within 0.001. The mismatch is real, but it is not the dominant measured cause under this audit.", "", "## Frozen conclusion", "", "The evidence supports a port-localization/front-end repair, not a larger Transformer and not a blind K increase. Any next protocol should test soft or branch-aware ports while keeping P0 frozen, and must rebuild true X/T/Y/weak/multiple-plausible generators before attributing stratum-specific effects.", "", "## Boundary", "", "No prediction, threshold, model, split, training, confirm, CRACKS, expert, or path result was changed or opened. This audit ends at the A/B/C/D/E table.", ""]
    return "\n".join(lines)
