#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Sequence

import numpy as np


DEVELOPMENT_SPLITS = frozenset({"train", "val"})


class JunctionBuildError(ValueError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((first[i] - second[i]) ** 2 for i in range(3)))


def normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        raise JunctionBuildError("zero_length_tangent")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def parse_line_ids(value: str) -> list[int]:
    ids = [int(part) for part in value.split("|") if part]
    if not ids:
        raise JunctionBuildError("no_incident_lines")
    return ids


def read_vtk(path: Path):
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader

    reader = vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfPoints() == 0:
        raise RuntimeError(f"No points in {path}")
    return polydata


def ordered_arm_point_ids(polydata, line_id: int, junction_point_id: int) -> list[int]:
    if line_id < 0 or line_id >= polydata.GetNumberOfCells():
        raise JunctionBuildError(f"line_id_out_of_range:{line_id}")
    cell = polydata.GetCell(line_id)
    point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
    if len(point_ids) < 2:
        raise JunctionBuildError(f"line_has_fewer_than_two_points:{line_id}")
    if point_ids[0] == junction_point_id:
        return point_ids
    if point_ids[-1] == junction_point_id:
        return list(reversed(point_ids))
    raise JunctionBuildError(
        f"junction_not_at_line_endpoint:point={junction_point_id},line={line_id}"
    )


def sample_polyline_at_arc(
    points: Sequence[Sequence[float]], arc_distance_mm: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    if arc_distance_mm < 0:
        raise ValueError("arc_distance_mm must be non-negative")
    travelled = 0.0
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        segment_length = distance(start, end)
        if segment_length <= 1e-12:
            continue
        if travelled + segment_length >= arc_distance_mm:
            fraction = (arc_distance_mm - travelled) / segment_length
            position = tuple(
                start[i] + fraction * (end[i] - start[i]) for i in range(3)
            )
            tangent = normalize(tuple(end[i] - start[i] for i in range(3)))
            return position, tangent, index  # type: ignore[return-value]
        travelled += segment_length
    raise JunctionBuildError(
        f"arc_out_of_range:available={travelled:.6f},required={arc_distance_mm:.6f}"
    )


def exposed_endpoint(
    points: Sequence[Sequence[float]], cut_distance_mm: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if cut_distance_mm <= 0:
        raise ValueError("cut_distance_mm must be positive")
    position, outward, _ = sample_polyline_at_arc(points, cut_distance_mm)
    return position, tuple(-value for value in outward)  # type: ignore[return-value]


def point_and_forward_tangent_at_arc(
    points: Sequence[Sequence[float]], arc_distance_mm: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position, tangent, _ = sample_polyline_at_arc(points, arc_distance_mm)
    return position, tangent


def decoy_arc_positions(
    length_mm: float, *, step_mm: float, endpoint_margin_mm: float
) -> list[float]:
    if step_mm <= 0:
        raise ValueError("step_mm must be positive")
    if endpoint_margin_mm < 0:
        raise ValueError("endpoint_margin_mm must be non-negative")
    last = length_mm - endpoint_margin_mm
    if last < endpoint_margin_mm:
        return []
    positions = []
    value = endpoint_margin_mm
    while value <= last + 1e-9:
        positions.append(value)
        value += step_mm
    return positions


def categorical_value_at_arc(
    points: Sequence[Sequence[float]], values: Sequence[int], arc_distance_mm: float
) -> int:
    if len(points) != len(values):
        raise ValueError("points and values must have equal length")
    if len(points) < 2:
        raise JunctionBuildError("line_has_fewer_than_two_points")
    _, _, segment_index = sample_polyline_at_arc(points, arc_distance_mm)
    return int(values[segment_index + 1])


def candidate_span(endpoints: Sequence[Sequence[float]]) -> float:
    if len(endpoints) < 2:
        raise JunctionBuildError("candidate_has_fewer_than_two_endpoints")
    return max(
        distance(endpoints[i], endpoints[j])
        for i in range(len(endpoints))
        for j in range(i + 1, len(endpoints))
    )


def junction_geometry(
    endpoints: Sequence[Sequence[float]], tangents: Sequence[Sequence[float]]
) -> dict[str, float]:
    x = np.asarray(endpoints, dtype=float)
    t = np.asarray(tangents, dtype=float)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    identity = np.eye(3)
    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    for point, tangent in zip(x, t):
        projection = identity - np.outer(tangent, tangent)
        matrix += projection
        vector += projection @ point
    center = np.linalg.pinv(matrix, rcond=1e-7) @ vector
    toward = center[None, :] - x
    forward = np.sum(toward * t, axis=1)
    residual = np.linalg.norm(toward - forward[:, None] * t, axis=1)
    center_distance = np.linalg.norm(toward, axis=1)
    alignment = forward / np.maximum(center_distance, 1e-12)
    return {
        "jg_span_mm": candidate_span(endpoints),
        "jg_residual_mean_mm": float(residual.mean()),
        "jg_residual_max_mm": float(residual.max()),
        "jg_forward_min_mm": float(forward.min()),
        "jg_forward_mean_mm": float(forward.mean()),
        "jg_alignment_min": float(alignment.min()),
        "jg_alignment_mean": float(alignment.mean()),
        "jg_center_distance_max_mm": float(center_distance.max()),
        "jg_center_distance_std_mm": float(center_distance.std()),
    }


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(distance(first, second) for first, second in zip(points[:-1], points[1:]))


def select_cut_distance(
    arm_lengths_mm: Sequence[float],
    *,
    target_distance_mm: float,
    max_arm_fraction: float,
) -> float:
    if not arm_lengths_mm or min(arm_lengths_mm) <= 0:
        raise JunctionBuildError("invalid_arm_length")
    if target_distance_mm <= 0:
        raise ValueError("target_distance_mm must be positive")
    if not 0 < max_arm_fraction < 1:
        raise ValueError("max_arm_fraction must be between 0 and 1")
    return min(target_distance_mm, max_arm_fraction * min(arm_lengths_mm))


def junction_key(row: dict[str, str]) -> str:
    return f"{row['scan_id']}:{row['side']}:{row['point_id']}"


def process_junction(
    row: dict[str, str],
    polydata,
    *,
    target_distance_mm: float,
    max_arm_fraction: float,
    max_span_mm: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    point_id = int(row["point_id"])
    degree = int(row["degree"])
    line_ids = parse_line_ids(row["incident_line_ids"])
    if degree not in {3, 4}:
        raise JunctionBuildError(f"unsupported_degree:{degree}")
    if len(line_ids) != degree:
        raise JunctionBuildError(
            f"degree_line_count_mismatch:degree={degree},lines={len(line_ids)}"
        )

    label_array = polydata.GetPointData().GetArray("segment_label")
    if label_array is None:
        raise JunctionBuildError("segment_label_missing")
    arms = []
    for arm_index, line_id in enumerate(line_ids):
        point_ids = ordered_arm_point_ids(polydata, line_id, point_id)
        points = [tuple(map(float, polydata.GetPoint(pid))) for pid in point_ids]
        labels = [int(label_array.GetTuple1(pid)) for pid in point_ids]
        arms.append((arm_index, line_id, points, labels, polyline_length(points)))
    arm_lengths = [arm_length for _, _, _, _, arm_length in arms]
    cut_distance_mm = select_cut_distance(
        arm_lengths,
        target_distance_mm=target_distance_mm,
        max_arm_fraction=max_arm_fraction,
    )

    key = junction_key(row)
    endpoints: list[tuple[float, float, float]] = []
    arm_rows: list[dict[str, object]] = []
    for arm_index, line_id, points, labels, arm_length_mm in arms:
        position, tangent = exposed_endpoint(points, cut_distance_mm)
        endpoint_label = categorical_value_at_arc(points, labels, cut_distance_mm)
        endpoints.append(position)
        arm_rows.append(
            {
                "junction_id": key,
                "scan_id": int(row["scan_id"]),
                "split": row["split"],
                "side": row["side"],
                "junction_point_id": point_id,
                "degree": degree,
                "arm_index": arm_index,
                "line_id": line_id,
                "arm_length_mm": arm_length_mm,
                "cut_distance_mm": cut_distance_mm,
                "endpoint_segment_label": endpoint_label,
                "endpoint_x_lps": position[0],
                "endpoint_y_lps": position[1],
                "endpoint_z_lps": position[2],
                "tangent_to_junction_x": tangent[0],
                "tangent_to_junction_y": tangent[1],
                "tangent_to_junction_z": tangent[2],
            }
        )

    span_mm = candidate_span(endpoints)
    return (
        {
            "junction_id": key,
            "scan_id": int(row["scan_id"]),
            "split": row["split"],
            "side": row["side"],
            "junction_point_id": point_id,
            "degree": degree,
            "cut_mode": "adaptive",
            "target_cut_distance_mm": target_distance_mm,
            "min_arm_length_mm": min(arm_lengths),
            "cut_distance_mm": cut_distance_mm,
            "n_endpoints": len(endpoints),
            "span_mm": span_mm,
            "build_success": 1,
            "span_eligible": int(span_mm <= max_span_mm),
            "failure_reason": "" if span_mm <= max_span_mm else "span_above_limit",
        },
        arm_rows,
    )


def failure_row(
    row: dict[str, str], target_distance_mm: float, reason: str
) -> dict[str, object]:
    return {
        "junction_id": junction_key(row),
        "scan_id": int(row["scan_id"]),
        "split": row["split"],
        "side": row["side"],
        "junction_point_id": int(row["point_id"]),
        "degree": int(row["degree"]),
        "cut_mode": "adaptive",
        "target_cut_distance_mm": target_distance_mm,
        "min_arm_length_mm": "",
        "cut_distance_mm": target_distance_mm,
        "n_endpoints": 0,
        "span_mm": "",
        "build_success": 0,
        "span_eligible": 0,
        "failure_reason": reason,
    }


def validate_source(source_root: Path) -> tuple[Path, Path]:
    metadata = source_root / "metadata"
    centerlines = source_root / "centerlines"
    required = [metadata / "cohort.csv", metadata / "junctions_truth.csv", centerlines]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source paths: " + ", ".join(missing))
    return metadata, centerlines


def development_junctions(metadata: Path) -> list[dict[str, str]]:
    cohort_rows = read_csv(metadata / "cohort.csv")
    split_by_scan = {int(row["scan_id"]): row["split"] for row in cohort_rows}
    rows = []
    for row in read_csv(metadata / "junctions_truth.csv"):
        scan_id = int(row["scan_id"])
        expected_split = split_by_scan.get(scan_id)
        if expected_split is None:
            raise ValueError(f"Junction scan absent from cohort.csv: {scan_id}")
        if row["split"] != expected_split:
            raise ValueError(
                f"Split mismatch for {scan_id}: {row['split']} != {expected_split}"
            )
        if expected_split in DEVELOPMENT_SPLITS:
            rows.append(row)
    return rows


def generate_positive_audit(
    *,
    source_root: Path,
    out_dir: Path,
    target_distance_mm: float,
    max_arm_fraction: float,
    max_span_mm: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata, centerlines = validate_source(source_root)
    junctions = development_junctions(metadata)
    audit_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    vtk_cache = {}

    for row in junctions:
        scan_id = int(row["scan_id"])
        side = row["side"]
        cache_key = (scan_id, side)
        if cache_key not in vtk_cache:
            path = centerlines / f"{scan_id}.coronary_{side}_centerline.vtk"
            vtk_cache[cache_key] = read_vtk(path)

        try:
            audit, arms = process_junction(
                row,
                vtk_cache[cache_key],
                target_distance_mm=target_distance_mm,
                max_arm_fraction=max_arm_fraction,
                max_span_mm=max_span_mm,
            )
        except JunctionBuildError as error:
            audit_rows.append(failure_row(row, target_distance_mm, str(error)))
            continue
        audit_rows.append(audit)
        arm_rows.extend(arms)

    if len(junctions) != 144:
        raise RuntimeError(f"Expected 144 train/val junctions, found {len(junctions)}")

    write_csv(out_dir / "junction_positive_audit_train_val.csv", audit_rows)
    write_csv(out_dir / "junction_positive_arms_train_val.csv", arm_rows)
    return audit_rows, arm_rows


def cell_point_ids(polydata, line_id: int) -> list[int]:
    if line_id < 0 or line_id >= polydata.GetNumberOfCells():
        raise JunctionBuildError(f"line_id_out_of_range:{line_id}")
    cell = polydata.GetCell(line_id)
    point_ids = [cell.GetPointId(index) for index in range(cell.GetNumberOfPoints())]
    if len(point_ids) < 2:
        raise JunctionBuildError(f"line_has_fewer_than_two_points:{line_id}")
    return point_ids


def candidate_endpoint_row(
    *,
    candidate_id: str,
    endpoint_index: int,
    source_role: str,
    source_line_id: int,
    position: Sequence[float],
    tangent: Sequence[float],
    source_arm_index: int | str,
    decoy_arc_mm: float | str = "",
    decoy_orientation: int | str = "",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "endpoint_index": endpoint_index,
        "source_role": source_role,
        "source_arm_index": source_arm_index,
        "source_line_id": source_line_id,
        "decoy_arc_mm": decoy_arc_mm,
        "decoy_orientation": decoy_orientation,
        "x_lps": position[0],
        "y_lps": position[1],
        "z_lps": position[2],
        "tangent_to_candidate_x": tangent[0],
        "tangent_to_candidate_y": tangent[1],
        "tangent_to_candidate_z": tangent[2],
    }


def generate_candidate_plan(
    *,
    source_root: Path,
    out_dir: Path,
    target_distance_mm: float,
    max_arm_fraction: float,
    max_span_mm: float,
    decoy_step_mm: float,
    decoy_endpoint_margin_mm: float,
    one_arm_per_positive: int,
    two_arm_per_positive: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    audit_rows, positive_arms = generate_positive_audit(
        source_root=source_root,
        out_dir=out_dir,
        target_distance_mm=target_distance_mm,
        max_arm_fraction=max_arm_fraction,
        max_span_mm=max_span_mm,
    )
    failed = [row for row in audit_rows if not int(row["build_success"])]
    ineligible = [row for row in audit_rows if not int(row["span_eligible"])]
    if failed or ineligible:
        raise RuntimeError(
            f"Positive protocol is incomplete: failed={len(failed)}, "
            f"span_ineligible={len(ineligible)}"
        )

    _, centerlines = validate_source(source_root)
    arms_by_junction: dict[str, list[dict[str, object]]] = {}
    for arm in positive_arms:
        arms_by_junction.setdefault(str(arm["junction_id"]), []).append(arm)

    vtk_cache = {}
    positive_objects = []
    negative_objects = []
    negative_index = 0

    for audit in audit_rows:
        junction_id = str(audit["junction_id"])
        arms = sorted(
            arms_by_junction[junction_id], key=lambda row: int(row["arm_index"])
        )
        degree = int(audit["degree"])
        scan_id = int(audit["scan_id"])
        side = str(audit["side"])
        split = str(audit["split"])
        positive_id = f"JPOS:{junction_id}"
        true_positions = [
            (
                float(arm["endpoint_x_lps"]),
                float(arm["endpoint_y_lps"]),
                float(arm["endpoint_z_lps"]),
            )
            for arm in arms
        ]
        true_tangents = [
            (
                float(arm["tangent_to_junction_x"]),
                float(arm["tangent_to_junction_y"]),
                float(arm["tangent_to_junction_z"]),
            )
            for arm in arms
        ]
        positive_row = {
            "candidate_id": positive_id,
            "source_junction_id": junction_id,
            "scan_id": scan_id,
            "split": split,
            "side": side,
            "degree": degree,
            "y": 1,
            "candidate_type": "positive_exact_arm_set",
            "n_replaced_arms": 0,
            "replaced_arm_indices": "",
            "decoy_line_ids": "",
            "decoy_arc_mm": "",
            "decoy_orientations": "",
            "span_mm": candidate_span(true_positions),
            "replaced_segment_labels": "",
            "decoy_segment_labels": "",
            "all_decoy_labels_different": "",
            "geometry_match_positive_id": positive_id,
            "geometry_match_distance": 0.0,
        }
        positive_row.update(junction_geometry(true_positions, true_tangents))
        positive_endpoints = []
        for index, (arm, position, tangent) in enumerate(
            zip(arms, true_positions, true_tangents)
        ):
            positive_endpoints.append(
                candidate_endpoint_row(
                    candidate_id=positive_id,
                    endpoint_index=index,
                    source_role="true_arm",
                    source_arm_index=int(arm["arm_index"]),
                    source_line_id=int(arm["line_id"]),
                    position=position,
                    tangent=tangent,
                )
            )
        positive_objects.append({"row": positive_row, "endpoints": positive_endpoints})

        cache_key = (scan_id, side)
        if cache_key not in vtk_cache:
            vtk_path = centerlines / f"{scan_id}.coronary_{side}_centerline.vtk"
            vtk_cache[cache_key] = read_vtk(vtk_path)
        polydata = vtk_cache[cache_key]
        label_array = polydata.GetPointData().GetArray("segment_label")
        if label_array is None:
            raise JunctionBuildError("segment_label_missing")
        incident_ids = {int(arm["line_id"]) for arm in arms}

        decoys = []
        for decoy_line_id in range(polydata.GetNumberOfCells()):
            if decoy_line_id in incident_ids:
                continue
            point_ids = cell_point_ids(polydata, decoy_line_id)
            points = [tuple(map(float, polydata.GetPoint(pid))) for pid in point_ids]
            point_labels = [int(label_array.GetTuple1(pid)) for pid in point_ids]
            length_mm = polyline_length(points)
            for arc_mm in decoy_arc_positions(
                length_mm,
                step_mm=decoy_step_mm,
                endpoint_margin_mm=decoy_endpoint_margin_mm,
            ):
                decoy_position, forward = point_and_forward_tangent_at_arc(points, arc_mm)
                decoy_label = categorical_value_at_arc(points, point_labels, arc_mm)
                if decoy_label <= 0:
                    continue
                if min(distance(decoy_position, point) for point in true_positions) > max_span_mm:
                    continue
                for orientation in (-1, 1):
                    decoys.append(
                        {
                            "line_id": decoy_line_id,
                            "arc_mm": arc_mm,
                            "orientation": orientation,
                            "position": decoy_position,
                            "tangent": tuple(orientation * value for value in forward),
                            "segment_label": decoy_label,
                            "physical_id": (decoy_line_id, round(arc_mm, 9)),
                        }
                    )

        replaced_labels = [int(arm["endpoint_segment_label"]) for arm in arms]
        seen = set()

        def add_negative(replaced_indices, assigned_decoys, candidate_type):
            nonlocal negative_index
            positions = list(true_positions)
            tangents = list(true_tangents)
            endpoint_tokens = [
                ("true", int(arm["line_id"]), int(arm["arm_index"])) for arm in arms
            ]
            for replaced_index, decoy in zip(replaced_indices, assigned_decoys):
                positions[replaced_index] = decoy["position"]
                tangents[replaced_index] = decoy["tangent"]
                endpoint_tokens[replaced_index] = (
                    "decoy",
                    decoy["line_id"],
                    round(decoy["arc_mm"], 9),
                    decoy["orientation"],
                )
            key = tuple(sorted(endpoint_tokens))
            if key in seen or candidate_span(positions) > max_span_mm:
                return
            seen.add(key)
            negative_index += 1
            candidate_id = f"JNEG:{negative_index:08d}"
            row = {
                "candidate_id": candidate_id,
                "source_junction_id": junction_id,
                "scan_id": scan_id,
                "split": split,
                "side": side,
                "degree": degree,
                "y": 0,
                "candidate_type": candidate_type,
                "n_replaced_arms": len(replaced_indices),
                "replaced_arm_indices": "|".join(map(str, replaced_indices)),
                "decoy_line_ids": "|".join(str(d["line_id"]) for d in assigned_decoys),
                "decoy_arc_mm": "|".join(f"{d['arc_mm']:.9f}" for d in assigned_decoys),
                "decoy_orientations": "|".join(
                    str(d["orientation"]) for d in assigned_decoys
                ),
                "span_mm": candidate_span(positions),
                "replaced_segment_labels": "|".join(
                    str(replaced_labels[index]) for index in replaced_indices
                ),
                "decoy_segment_labels": "|".join(
                    str(d["segment_label"]) for d in assigned_decoys
                ),
                "all_decoy_labels_different": int(
                    all(
                        replaced_labels[index] != decoy["segment_label"]
                        for index, decoy in zip(replaced_indices, assigned_decoys)
                    )
                ),
                "geometry_match_positive_id": positive_id,
                "geometry_match_distance": "",
            }
            row.update(junction_geometry(positions, tangents))
            endpoints = []
            replacement_by_index = dict(zip(replaced_indices, assigned_decoys))
            for endpoint_index in range(degree):
                if endpoint_index in replacement_by_index:
                    decoy = replacement_by_index[endpoint_index]
                    endpoints.append(
                        candidate_endpoint_row(
                            candidate_id=candidate_id,
                            endpoint_index=endpoint_index,
                            source_role="decoy",
                            source_arm_index="",
                            source_line_id=decoy["line_id"],
                            position=decoy["position"],
                            tangent=decoy["tangent"],
                            decoy_arc_mm=decoy["arc_mm"],
                            decoy_orientation=decoy["orientation"],
                        )
                    )
                else:
                    arm = arms[endpoint_index]
                    endpoints.append(
                        candidate_endpoint_row(
                            candidate_id=candidate_id,
                            endpoint_index=endpoint_index,
                            source_role="true_arm",
                            source_arm_index=int(arm["arm_index"]),
                            source_line_id=int(arm["line_id"]),
                            position=true_positions[endpoint_index],
                            tangent=true_tangents[endpoint_index],
                        )
                    )
            negative_objects.append({"row": row, "endpoints": endpoints})

        for replaced_index in range(degree):
            for decoy in decoys:
                add_negative(
                    (replaced_index,),
                    (decoy,),
                    "negative_one_arm_replacement",
                )

        for replaced_indices in itertools.combinations(range(degree), 2):
            for first, second in itertools.combinations(decoys, 2):
                if first["physical_id"] == second["physical_id"]:
                    continue
                assignments = ((first, second), (second, first))
                assigned = max(
                    assignments,
                    key=lambda values: sum(
                        replaced_labels[index] != decoy["segment_label"]
                        for index, decoy in zip(replaced_indices, values)
                    ),
                )
                add_negative(
                    replaced_indices,
                    assigned,
                    "negative_two_arm_replacement",
                )

    feature_names = [
        "jg_span_mm",
        "jg_residual_mean_mm",
        "jg_residual_max_mm",
        "jg_forward_min_mm",
        "jg_forward_mean_mm",
        "jg_alignment_min",
        "jg_alignment_mean",
        "jg_center_distance_max_mm",
        "jg_center_distance_std_mm",
    ]
    positive_by_id = {obj["row"]["candidate_id"]: obj for obj in positive_objects}
    train_degree3 = [
        obj["row"]
        for obj in positive_objects
        if obj["row"]["split"] == "train" and obj["row"]["degree"] == 3
    ]
    train_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in train_degree3]
    )
    scales = train_matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    for obj in negative_objects:
        row = obj["row"]
        positive = positive_by_id[row["geometry_match_positive_id"]]["row"]
        vector = np.asarray([float(row[name]) for name in feature_names])
        reference = np.asarray([float(positive[name]) for name in feature_names])
        row["geometry_match_distance"] = float(
            np.linalg.norm((vector - reference) / scales)
        )

    negatives_by_junction: dict[str, list[dict[str, object]]] = {}
    for obj in negative_objects:
        negatives_by_junction.setdefault(
            str(obj["row"]["source_junction_id"]), []
        ).append(obj)

    def selection_key(obj):
        return (
            -obj["row"]["all_decoy_labels_different"],
            obj["row"]["geometry_match_distance"],
            obj["row"]["candidate_id"],
        )

    primary_positives = [obj for obj in positive_objects if obj["row"]["degree"] == 3]
    selected_objects = list(primary_positives)
    selection_audit = []
    for positive in primary_positives:
        row = positive["row"]
        junction_id = str(row["source_junction_id"])
        pool = negatives_by_junction.get(junction_id, [])
        one = sorted(
            [obj for obj in pool if obj["row"]["n_replaced_arms"] == 1],
            key=selection_key,
        )
        two = sorted(
            [obj for obj in pool if obj["row"]["n_replaced_arms"] == 2],
            key=selection_key,
        )
        chosen = one[:one_arm_per_positive] + two[:two_arm_per_positive]
        selected_objects.extend(chosen)
        selection_audit.append(
            {
                "junction_id": junction_id,
                "scan_id": row["scan_id"],
                "split": row["split"],
                "raw_one_arm": len(one),
                "raw_two_arm": len(two),
                "selected_one_arm": min(len(one), one_arm_per_positive),
                "selected_two_arm": min(len(two), two_arm_per_positive),
            }
        )

    plan_rows = [obj["row"] for obj in selected_objects]
    endpoint_rows = [endpoint for obj in selected_objects for endpoint in obj["endpoints"]]
    write_csv(out_dir / "junction_relation_plan_train_val.csv", plan_rows)
    write_csv(out_dir / "junction_candidate_endpoints_train_val.csv", endpoint_rows)
    write_csv(out_dir / "junction_selection_audit_train_val.csv", selection_audit)
    return audit_rows, plan_rows, endpoint_rows, len(negative_objects)