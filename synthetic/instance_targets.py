"""Rasterize separate observed and latent structural targets."""

from __future__ import annotations

import math

import numpy as np

from .geometry_generator import GeometrySample


def _sample_polyline(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        count = max(2, int(math.ceil(float(np.linalg.norm(delta)) * 2.0)))
        t = np.linspace(0.0, 1.0, count, endpoint=False, dtype=np.float32)[:, None]
        samples.append(start[None] * (1.0 - t) + end[None] * t)
        angles.append(np.full(count, math.atan2(float(delta[1]), float(delta[0])), dtype=np.float32))
    samples.append(points[-1:])
    angles.append(angles[-1][-1:])
    return np.concatenate(samples), np.concatenate(angles)


def _paint_disks(mask: np.ndarray, points_xy: np.ndarray, radius: int, value: int | bool = True) -> None:
    height, width = mask.shape
    for x_float, y_float in points_xy:
        x, y = int(round(float(x_float))), int(round(float(y_float)))
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        selected = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        mask[y0:y1, x0:x1][selected] = value


def rasterize_targets(geometry: GeometrySample, image_size: int, line_radius: int = 1) -> dict[str, np.ndarray | list]:
    height = width = int(image_size)
    branch_ids = sorted(branch.branch_id for branch in geometry.branches)
    instance_ids = sorted({branch.instance_id for branch in geometry.branches})
    branch_index = {value: index for index, value in enumerate(branch_ids)}
    instance_index = {value: index for index, value in enumerate(instance_ids)}
    branch_by_id = {branch.branch_id: branch for branch in geometry.branches}
    branch_masks = np.zeros((len(branch_ids), height, width), dtype=bool)
    branch_centerlines = np.zeros_like(branch_masks)
    branch_cos2 = np.zeros((len(branch_ids), height, width), dtype=np.float32)
    branch_sin2 = np.zeros_like(branch_cos2)
    latent_instance_masks = np.zeros((len(instance_ids), height, width), dtype=bool)
    endpoint_map = np.zeros((height, width), dtype=bool)

    for branch in geometry.branches:
        index = branch_index[branch.branch_id]
        points, angles = _sample_polyline(branch.points_xy)
        _paint_disks(branch_masks[index], points, line_radius)
        _paint_disks(branch_centerlines[index], points, 0)
        for point, angle in zip(points, angles):
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if 0 <= y < height and 0 <= x < width:
                branch_cos2[index, y, x] = math.cos(2.0 * float(angle))
                branch_sin2[index, y, x] = math.sin(2.0 * float(angle))
        latent_instance_masks[instance_index[branch.instance_id]] |= branch_masks[index]
        _paint_disks(endpoint_map, branch.points_xy[[0, -1]], 1)

    visible_fault_mask = np.any(latent_instance_masks, axis=0)
    visible_centerline_map = np.any(branch_centerlines, axis=0)
    positive_gap_mask = np.zeros((height, width), dtype=bool)
    positive_gap_centerline = np.zeros((height, width), dtype=bool)
    negative_gap_mask = np.zeros((height, width), dtype=bool)
    positive_gap_owner = np.zeros((height, width), dtype=np.int16)
    positive_gap_masks: list[np.ndarray] = []
    negative_gap_masks: list[np.ndarray] = []
    gap_records: list[dict[str, object]] = []
    for gap in geometry.gaps:
        points, _ = _sample_polyline(gap.points_xy)
        candidate = np.zeros((height, width), dtype=bool)
        candidate_centerline = np.zeros_like(candidate)
        _paint_disks(candidate, points, line_radius)
        _paint_disks(candidate_centerline, points, 0)
        if gap.gap_type == "positive":
            if gap.latent_instance_id not in instance_index:
                raise ValueError("Positive gap must name one existing latent instance")
            candidate &= ~visible_fault_mask
            candidate_centerline &= candidate
            positive_gap_mask |= candidate
            positive_gap_centerline |= candidate_centerline
            owner_index = instance_index[int(gap.latent_instance_id)]
            latent_instance_masks[owner_index] |= candidate
            positive_gap_owner[candidate] = int(gap.latent_instance_id)
            positive_gap_masks.append(candidate.copy())
            _paint_disks(endpoint_map, gap.points_xy[[0, -1]], 3, value=False)
        elif gap.gap_type == "negative":
            candidate &= ~np.any(latent_instance_masks, axis=0)
            negative_gap_mask |= candidate
            negative_gap_masks.append(candidate.copy())
        else:
            raise ValueError(f"Unknown gap type: {gap.gap_type}")
        gap_records.append(
            {
                "gap_type": gap.gap_type,
                "latent_instance_id": gap.latent_instance_id,
                "pixel_count": int(candidate.sum()),
                "endpoint_xy": gap.points_xy[[0, -1]].astype(float).tolist(),
            }
        )

    latent_fault_mask = np.any(latent_instance_masks, axis=0)
    gap_mask = latent_fault_mask & ~visible_fault_mask
    if not np.array_equal(gap_mask, positive_gap_mask):
        raise AssertionError("Positive gap must equal latent_fault_mask & ~visible_fault_mask")
    latent_centerline_map = visible_centerline_map | positive_gap_centerline

    junction_map = np.zeros((height, width), dtype=bool)
    junction_points = np.asarray([junction.point_xy for junction in geometry.junctions], dtype=np.float32)
    if len(junction_points):
        _paint_disks(junction_map, junction_points, 2)
        _paint_disks(endpoint_map, junction_points, 3, value=False)

    instance_overlap = latent_instance_masks.sum(axis=0) > 1
    instance_visualization_map = np.zeros((height, width), dtype=np.int16)
    for index, instance_id in enumerate(instance_ids):
        selected = latent_instance_masks[index] & (instance_visualization_map == 0)
        instance_visualization_map[selected] = instance_id
    branch_visualization_map = np.zeros((height, width), dtype=np.int16)
    for index, branch_id in enumerate(branch_ids):
        selected = branch_centerlines[index] & (branch_visualization_map == 0)
        branch_visualization_map[selected] = branch_id

    relation_matrix = np.zeros((len(branch_ids), len(branch_ids)), dtype=bool)
    eligible_relation_matrix = np.zeros_like(relation_matrix)
    junction_records = []
    for junction in geometry.junctions:
        incident_indices = [branch_index[value] for value in junction.incident_branch_ids]
        for first_position, first in enumerate(incident_indices):
            for second in incident_indices[first_position + 1 :]:
                eligible_relation_matrix[first, second] = True
                eligible_relation_matrix[second, first] = True
        for first, second in junction.continuation_relation:
            relation_matrix[branch_index[first], branch_index[second]] = True
            relation_matrix[branch_index[second], branch_index[first]] = True
        junction_records.append(
            {
                "junction_type": junction.junction_type,
                "point_xy": [float(junction.point_xy[0]), float(junction.point_xy[1])],
                "incident_branch_ids": list(junction.incident_branch_ids),
                "incident_instance_ids": list(junction.incident_instance_ids),
                "continuation_relation": [list(pair) for pair in junction.continuation_relation],
            }
        )
    return {
        "visible_fault_mask": visible_fault_mask,
        "latent_fault_mask": latent_fault_mask,
        "gap_mask": gap_mask,
        "positive_gap_mask": positive_gap_mask,
        "negative_gap_mask": negative_gap_mask,
        "positive_gap_masks": np.stack(positive_gap_masks) if positive_gap_masks else np.zeros((0, height, width), dtype=bool),
        "negative_gap_masks": np.stack(negative_gap_masks) if negative_gap_masks else np.zeros((0, height, width), dtype=bool),
        "positive_gap_owner": positive_gap_owner,
        "visible_centerline_map": visible_centerline_map,
        "latent_centerline_map": latent_centerline_map,
        "instance_visualization_map": instance_visualization_map,
        "instance_masks": latent_instance_masks,
        "instance_overlap_mask": instance_overlap,
        "branch_visualization_map": branch_visualization_map,
        "branch_masks": branch_masks,
        "branch_centerlines": branch_centerlines,
        "branch_tangent_cos2": branch_cos2,
        "branch_tangent_sin2": branch_sin2,
        "junction_map": junction_map,
        "endpoint_map": endpoint_map,
        "continuation_relation_matrix": relation_matrix,
        "continuation_eligible_matrix": eligible_relation_matrix,
        "branch_ids": branch_ids,
        "branch_instance_id": [branch_by_id[value].instance_id for value in branch_ids],
        "fault_instance_ids": instance_ids,
        "junctions": junction_records,
        "gaps": gap_records,
        "strata": list(geometry.strata),
    }
