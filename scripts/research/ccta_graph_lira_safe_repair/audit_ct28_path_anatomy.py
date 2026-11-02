#!/usr/bin/env python3
"""Compare frozen validation routes with their intended source centerline arcs.

This is post-hoc evaluation. VTK centerlines never enter CT path construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from build_junction_plan import ordered_arm_point_ids, read_vtk
from ct28_path_stage import voxel_to_lps
import run_ct28_graph_lira as graph


ROOT = graph.REPO_ROOT
SCENES = ROOT / "artifacts/varvara/ct28_graph_lira"
JUNCTION = ROOT / "artifacts/varvara/ct28_junction"
SOURCE = ROOT / "varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack"
OUTPUT = ROOT / "varvara/anza_lira_local_results/ct28_path_protocol_v1"
GEOMETRY = ROOT / "artifacts/varvara/ct28_pair/expected_ct_geometry.csv"
VAL_IDS = {957, 961, 965, 966, 974}
TEST_IDS = {954, 958, 972, 973, 980, 984}
MATCH_TOL_MM = 1e-4
SAMPLE_STEP_MM = 0.25
EVAL_TOL_MM = 0.75


def dense_polyline(points: np.ndarray, step_mm: float = SAMPLE_STEP_MM) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("reference polyline must contain at least two points")
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        if length <= 1e-10:
            continue
        count = max(1, int(np.ceil(length / step_mm)))
        pieces.extend(start + fraction * (end - start)
                      for fraction in np.linspace(0.0, 1.0, count, endpoint=False))
    pieces.append(points[-1])
    return np.asarray(pieces, dtype=float)


def affine_from_row(row: pd.Series) -> np.ndarray:
    affine = np.eye(4)
    for i in range(3):
        for j in range(4):
            affine[i, j] = float(row[f"affine_{i}{j}"])
    return affine


def load_vtk(patient: int, side: str, cache: dict) -> dict:
    key = (patient, side)
    if key not in cache:
        path = SOURCE / "centerlines" / f"{patient}.coronary_{side}_centerline.vtk"
        poly = read_vtk(path)
        points = np.asarray([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())], dtype=float)
        owners: dict[int, set[int]] = {}
        lines = {}
        for line_id in range(poly.GetNumberOfCells()):
            cell = poly.GetCell(line_id)
            ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
            lines[line_id] = ids
            for point_id in ids:
                owners.setdefault(point_id, set()).add(line_id)
        cache[key] = {"poly": poly, "points": points, "tree": cKDTree(points),
                      "owners": owners, "lines": lines}
    return cache[key]


def pair_reference(patient: int, positions: np.ndarray, cache: dict) -> tuple[str, int, np.ndarray]:
    matches = []
    for side in ("left", "right"):
        vtk = load_vtk(patient, side, cache)
        distances, point_ids = vtk["tree"].query(positions)
        if float(np.max(distances)) > MATCH_TOL_MM:
            continue
        common = vtk["owners"][int(point_ids[0])] & vtk["owners"][int(point_ids[1])]
        for line_id in common:
            ids = vtk["lines"][line_id]
            first, second = ids.index(int(point_ids[0])), ids.index(int(point_ids[1]))
            if first == second:
                continue
            ordered = ids[first:second + 1] if first < second else list(reversed(ids[second:first + 1]))
            matches.append((side, line_id, dense_polyline(vtk["points"][ordered])))
    if len(matches) != 1:
        raise ValueError(f"PAIR reference is not uniquely identified: patient={patient}, matches={len(matches)}")
    if np.linalg.norm(matches[0][2][0] - positions[0]) > MATCH_TOL_MM or \
            np.linalg.norm(matches[0][2][-1] - positions[1]) > MATCH_TOL_MM:
        raise ValueError("PAIR reference does not start/end at the selected endpoints")
    return matches[0]


def junction_references(patient: int, source_id: str, positions: np.ndarray,
                        positives: pd.DataFrame, truth: pd.DataFrame,
                        cache: dict) -> list[tuple[str, int, np.ndarray]]:
    scan, side, point = source_id.split(":")
    if int(scan) != patient:
        raise ValueError("JUNCTION source patient differs from scene patient")
    row = truth.loc[(patient, side, int(point))]
    incident = {int(value) for value in str(row.incident_line_ids).split("|")}
    positive = positives[positives.candidate_id.eq(f"JPOS:{source_id}")]
    if len(positive) != 3:
        raise ValueError("JUNCTION lacks three original positive endpoints")
    true_positions = positive[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
    separation = np.linalg.norm(positions[:, None, :] - true_positions[None, :, :], axis=2)
    assignment = np.argmin(separation, axis=1)
    if (float(separation[np.arange(3), assignment].max()) > MATCH_TOL_MM or
            len(set(assignment)) != 3):
        raise ValueError("JUNCTION scene endpoints do not uniquely match true exposed arms")
    vtk = load_vtk(patient, side, cache)
    references = []
    for arm, raw_index in enumerate(assignment):
        line_id = int(positive.iloc[int(raw_index)].source_line_id)
        if line_id not in incident:
            raise ValueError("JUNCTION source arm is not incident to annotated branch point")
        point_ids = ordered_arm_point_ids(vtk["poly"], line_id, int(point))
        line_points = vtk["points"][point_ids]
        endpoint = positions[arm]
        candidates = []
        for index, (start, end) in enumerate(zip(line_points[:-1], line_points[1:])):
            direction = end - start
            denominator = float(direction @ direction)
            if denominator <= 1e-12:
                continue
            fraction = float(np.clip(((endpoint - start) @ direction) / denominator, 0, 1))
            projected = start + fraction * direction
            candidates.append((float(np.linalg.norm(projected - endpoint)), index, fraction))
        if not candidates:
            raise ValueError("empty incident arm")
        distance, index, _ = min(candidates)
        if distance > MATCH_TOL_MM:
            raise ValueError("exposed JUNCTION endpoint is not on its annotated arm")
        arc = np.vstack((line_points[:index + 1], endpoint))[::-1]
        sampled = dense_polyline(arc)
        true_center = row[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
        if (np.linalg.norm(sampled[0] - endpoint) > MATCH_TOL_MM or
                np.linalg.norm(sampled[-1] - true_center) > MATCH_TOL_MM):
            raise ValueError("JUNCTION reference does not connect exposed arm to true center")
        references.append((side, line_id, sampled))
    if {line_id for _, line_id, _ in references} != incident:
        raise ValueError("JUNCTION arm set differs from annotated incident lines")
    return references


def distance_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    to_ref = cKDTree(reference).query(predicted)[0]
    to_pred = cKDTree(predicted).query(reference)[0]
    predicted_length = float(np.linalg.norm(np.diff(predicted, axis=0), axis=1).sum())
    reference_length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
    return {
        "route_length_mm": predicted_length,
        "true_arc_length_mm": reference_length,
        "route_over_true_arc_length": predicted_length / max(reference_length, 1e-6),
        "route_to_true_mean_mm": float(to_ref.mean()),
        "route_to_true_p95_mm": float(np.quantile(to_ref, 0.95)),
        "route_to_true_max_mm": float(to_ref.max()),
        "route_within_0p75mm_fraction": float(np.mean(to_ref <= EVAL_TOL_MM)),
        "route_within_1p5mm_fraction": float(np.mean(to_ref <= 1.5)),
        "true_to_route_mean_mm": float(to_pred.mean()),
        "true_to_route_p95_mm": float(np.quantile(to_pred, 0.95)),
        "true_covered_0p75mm_fraction": float(np.mean(to_pred <= EVAL_TOL_MM)),
        "true_covered_1p5mm_fraction": float(np.mean(to_pred <= 1.5)),
    }


def other_line_distance(patient: int, side: str, selected: set[int],
                        predicted: np.ndarray, cache: dict) -> tuple[float, int | None, float, int | None]:
    vtk = load_vtk(patient, side, cache)
    if len(predicted) < 3:
        return float("nan"), None, float("nan"), None
    closest = (float("inf"), None)
    closest_nonadjacent = (float("inf"), None)
    selected_points = {point_id for line_id in selected for point_id in vtk["lines"][line_id]}
    for line_id, ids in vtk["lines"].items():
        if line_id in selected or len(ids) < 2:
            continue
        points = dense_polyline(vtk["points"][ids])
        distance = float(np.min(cKDTree(points).query(predicted[1:-1])[0]))
        if distance < closest[0]:
            closest = distance, line_id
        if selected_points.isdisjoint(ids) and distance < closest_nonadjacent[0]:
            closest_nonadjacent = distance, line_id
    return closest[0], closest[1], closest_nonadjacent[0], closest_nonadjacent[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    route_path = args.path_dir / "paths_val.jsonl"
    routes = [json.loads(line) for line in route_path.open()]
    path_metrics = pd.read_csv(args.path_dir / "path_metrics_val.csv")
    plans = pd.read_csv(SCENES / "scene_plan_train_val.csv").set_index("scene_id")
    endpoints = pd.read_csv(SCENES / "scene_endpoints_train_val.csv")
    candidates = pd.read_csv(SCENES / "scene_candidates_train_val.csv")
    positives = pd.read_csv(JUNCTION / "junction_candidate_endpoints_train_val.csv")
    truth = pd.read_csv(SOURCE / "metadata/junctions_truth.csv")
    truth = truth[truth.split.eq("val")].set_index(["scan_id", "side", "point_id"])
    geometry = pd.read_csv(GEOMETRY).set_index("scan_id")
    if len(routes) != 52 or {int(row["scan_id"]) for row in routes} != VAL_IDS:
        raise ValueError("anatomy audit must use the frozen 52 accepted validation routes")
    expected_scenes = set(path_metrics.loc[path_metrics.path_attempted, "scene_id"])
    if expected_scenes != {row["scene_id"] for row in routes} or int(path_metrics.accepted.sum()) != 52:
        raise ValueError("saved routes differ from accepted structural validation scenes")
    cache, rows = {}, []
    for scene in routes:
        patient = int(scene["scan_id"])
        if patient in TEST_IDS or plans.loc[scene["scene_id"]].split != "val":
            raise ValueError("held-out or non-validation scene entered anatomy audit")
        affine = affine_from_row(geometry.loc[patient])
        scene_endpoints = endpoints[endpoints.scene_id.eq(scene["scene_id"])].set_index("endpoint_id")
        scene_candidates = candidates[candidates.scene_id.eq(scene["scene_id"])].set_index("candidate_id")
        if len(scene["routes"]) != len(scene["selected_candidate_ids"]):
            raise ValueError("route/candidate count differs")
        components = []
        for candidate_id, route in zip(scene["selected_candidate_ids"], scene["routes"]):
            candidate = scene_candidates.loc[candidate_id]
            if int(candidate.y) != 1 or str(candidate.candidate_type) != route["kind"]:
                raise ValueError("frozen accepted route is not the true selected relation")
            endpoint_ids = str(candidate.endpoint_ids).split("|")
            positions = scene_endpoints.loc[endpoint_ids][["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
            if route["kind"] == "PAIR":
                references = [pair_reference(patient, positions, cache)]
            elif route["kind"] == "JUNCTION":
                source_id = str(plans.loc[scene["scene_id"]].junction_source_id)
                references = junction_references(patient, source_id, positions, positives, truth, cache)
            else:
                raise ValueError("unsupported route kind")
            if len(route["paths_voxel"]) != len(references):
                raise ValueError("route/reference arm count differs")
            components.append((candidate_id, route, references))
        for candidate_id, route, references in components:
            component_lines = {(side, line_id) for side, line_id, _ in references}
            for arm_index, (path, (side, line_id, reference)) in enumerate(zip(route["paths_voxel"], references)):
                predicted = voxel_to_lps(np.asarray(path, dtype=float), affine)
                if len(predicted) < 2:
                    raise ValueError("degenerate saved route entered anatomy audit")
                selected_same_side = {line for selected_side, line in component_lines if selected_side == side}
                other_distance, other_line_id, nonadjacent_distance, nonadjacent_line_id = other_line_distance(
                    patient, side, selected_same_side, predicted, cache)
                rows.append({
                    "scene_id": scene["scene_id"], "scan_id": patient,
                    "scene_family": str(plans.loc[scene["scene_id"]].scene_family),
                    "candidate_id": candidate_id, "kind": route["kind"],
                    "arm_index": arm_index, "side": side, "true_line_id": line_id,
                    "route_voxels": len(predicted), "reference_samples": len(reference),
                    **distance_metrics(predicted, reference),
                    "other_line_min_distance_mm": other_distance,
                    "other_line_id": other_line_id,
                    "nonadjacent_line_min_distance_mm": nonadjacent_distance,
                    "nonadjacent_line_id": nonadjacent_line_id,
                })
    frame = pd.DataFrame(rows)
    if (len(frame) != 62 or frame.scene_id.nunique() != 52 or
            int(frame.kind.eq("PAIR").sum()) != 50 or int(frame.kind.eq("JUNCTION").sum()) != 12):
        raise ValueError("reference mapping did not cover all accepted PAIR/JUNCTION arms")
    frame.to_csv(args.path_dir / "path_anatomy_metrics_val.csv", index=False)
    summary = {
        "status": "validation_posthoc_anatomy_audit",
        "held_out_test_accessed": False,
        "source_paths_sha256": graph.sha256(route_path),
        "source_scene_endpoints_sha256": graph.sha256(SCENES / "scene_endpoints_train_val.csv"),
        "source_junction_truth_sha256": graph.sha256(SOURCE / "metadata/junctions_truth.csv"),
        "scenes": int(frame.scene_id.nunique()), "pair_routes": 50,
        "junction_components": 4, "junction_arms": 12,
        "evaluation_tolerance_mm": EVAL_TOL_MM,
        "by_kind": frame.groupby("kind").agg(
            arms=("arm_index", "size"),
            route_within_0p75_mean=("route_within_0p75mm_fraction", "mean"),
            route_within_1p5_mean=("route_within_1p5mm_fraction", "mean"),
            true_covered_0p75_mean=("true_covered_0p75mm_fraction", "mean"),
            true_covered_1p5_mean=("true_covered_1p5mm_fraction", "mean"),
            route_over_true_arc_length_median=("route_over_true_arc_length", "median"),
            route_over_true_arc_length_max=("route_over_true_arc_length", "max"),
            route_to_true_p95_median_mm=("route_to_true_p95_mm", "median"),
            route_to_true_p95_max_mm=("route_to_true_p95_mm", "max"),
            other_line_proximity_0p75_count=("other_line_min_distance_mm",
                                               lambda values: int((values <= EVAL_TOL_MM).sum())),
            nonadjacent_line_proximity_0p75_count=("nonadjacent_line_min_distance_mm",
                                                   lambda values: int((values <= EVAL_TOL_MM).sum())),
        ).to_dict(orient="index"),
        "by_patient": frame.groupby("scan_id").agg(
            scenes=("scene_id", "nunique"),
            arms=("arm_index", "size"),
            route_within_0p75_mean=("route_within_0p75mm_fraction", "mean"),
            true_covered_0p75_mean=("true_covered_0p75mm_fraction", "mean"),
        ).to_dict(orient="index"),
        "by_scene_family": frame.groupby("scene_family").agg(
            scenes=("scene_id", "nunique"),
            arms=("arm_index", "size"),
            route_within_0p75_mean=("route_within_0p75mm_fraction", "mean"),
            true_covered_0p75_mean=("true_covered_0p75mm_fraction", "mean"),
        ).to_dict(orient="index"),
        "largest_route_deviations": frame.nlargest(5, "route_to_true_p95_mm")[[
            "scene_id", "candidate_id", "arm_index", "route_to_true_p95_mm",
            "route_over_true_arc_length"
        ]].to_dict(orient="records"),
    }
    (args.path_dir / "anatomy_summary_val.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
