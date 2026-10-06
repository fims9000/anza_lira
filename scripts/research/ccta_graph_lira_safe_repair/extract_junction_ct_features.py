#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/research/ccta_graph_lira_safe_repair/junction_ct_features.json"
)
GEOMETRY_FEATURES = (
    "jg_span_mm",
    "jg_residual_mean_mm",
    "jg_residual_max_mm",
    "jg_forward_min_mm",
    "jg_forward_mean_mm",
    "jg_alignment_min",
    "jg_alignment_mean",
    "jg_center_distance_max_mm",
    "jg_center_distance_std_mm",
)
METADATA_COLUMNS = (
    "candidate_id",
    "source_junction_id",
    "scan_id",
    "split",
    "side",
    "degree",
    "y",
    "benchmark_group",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def normalize(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(value))
    if length <= 1e-12:
        raise ValueError("zero-length direction")
    return value / length


def orthobasis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    direction = normalize(direction)
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    first = normalize(np.cross(direction, reference))
    second = normalize(np.cross(direction, first))
    return first, second


def estimate_junction_center(
    endpoints: np.ndarray, tangents: np.ndarray
) -> np.ndarray:
    endpoints = np.asarray(endpoints, dtype=float)
    tangents = np.array(tangents, dtype=float, copy=True)
    if endpoints.shape != (3, 3) or tangents.shape != (3, 3):
        raise ValueError("degree-3 endpoints and tangents must have shape (3, 3)")
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    identity = np.eye(3)
    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    for point, tangent in zip(endpoints, tangents):
        projection = identity - np.outer(tangent, tangent)
        matrix += projection
        vector += projection @ point
    return np.linalg.pinv(matrix, rcond=1e-7) @ vector


def candidate_sample_points(
    endpoints: np.ndarray,
    tangents: np.ndarray,
    *,
    profile_n: int,
    ring_n: int,
    radii_mm: Sequence[float],
    longitudinal_range: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if profile_n < 2 or ring_n < 3:
        raise ValueError("profile_n and ring_n are too small")
    if len(longitudinal_range) != 2:
        raise ValueError("longitudinal_range must contain two values")
    start, stop = map(float, longitudinal_range)
    if not 0 <= start < stop <= 1:
        raise ValueError("longitudinal_range must lie within [0, 1]")
    radii = np.asarray(radii_mm, dtype=float)
    if len(radii) == 0 or np.any(radii <= 0):
        raise ValueError("radii_mm must be positive")

    center = estimate_junction_center(endpoints, tangents)
    channels = 1 + ring_n * len(radii)
    points = np.empty((3, profile_n, channels, 3), dtype=float)
    profile_t = np.linspace(start, stop, profile_n)
    angles = np.linspace(0.0, 2.0 * math.pi, ring_n, endpoint=False)
    for arm_index, endpoint in enumerate(np.asarray(endpoints, dtype=float)):
        direction = normalize(center - endpoint)
        first, second = orthobasis(direction)
        axis = endpoint[None, :] + profile_t[:, None] * (center - endpoint)[None, :]
        points[arm_index, :, 0, :] = axis
        offset = 1
        for radius in radii:
            ring_offsets = np.stack(
                [
                    radius * (math.cos(angle) * first + math.sin(angle) * second)
                    for angle in angles
                ]
            )
            points[arm_index, :, offset : offset + ring_n, :] = (
                axis[:, None, :] + ring_offsets[None, :, :]
            )
            offset += ring_n
    return center, points


def sample_candidate_tensor(
    endpoints: np.ndarray,
    tangents: np.ndarray,
    ct: np.ndarray,
    inv_affine: np.ndarray,
    sample_world: Callable,
    *,
    profile_n: int,
    ring_n: int,
    radii_mm: Sequence[float],
    longitudinal_range: Sequence[float],
    validate_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    center, points = candidate_sample_points(
        endpoints,
        tangents,
        profile_n=profile_n,
        ring_n=ring_n,
        radii_mm=radii_mm,
        longitudinal_range=longitudinal_range,
    )
    if validate_bounds:
        ras_points = points.copy()
        ras_points[..., :2] *= -1.0
        flat = ras_points.reshape(-1, 3)
        voxels = (
            inv_affine @ np.c_[flat, np.ones(len(flat), dtype=float)].T
        ).T[:, :3]
        upper = np.asarray(ct.shape, dtype=float) - 1.0
        if not bool(((voxels >= 0.0) & (voxels <= upper)).all()):
            raise ValueError("candidate sampling grid leaves the CT volume")
    values = sample_world(ct, inv_affine, points)
    return center, values.astype(np.float32)


def clip_hu(values: np.ndarray, hu_clip: Sequence[float]) -> np.ndarray:
    if len(hu_clip) != 2 or float(hu_clip[0]) >= float(hu_clip[1]):
        raise ValueError("hu_clip must contain increasing lower and upper bounds")
    return np.clip(values, float(hu_clip[0]), float(hu_clip[1])).astype(np.float32)


def arm_feature_names(radii_mm: Sequence[float]) -> list[str]:
    names = [
        "axis_mean",
        "axis_std",
        "axis_min",
        "axis_p10",
        "axis_median",
        "axis_p90",
        "axis_max",
        "axis_frac_gt150",
        "axis_frac_gt250",
        "axis_mean_abs_diff",
    ]
    for radius in radii_mm:
        radius_name = f"r{float(radius):g}mm"
        names.extend(
            [
                f"{radius_name}_ring_mean",
                f"{radius_name}_ring_std",
                f"{radius_name}_contrast_mean",
                f"{radius_name}_contrast_min",
                f"{radius_name}_contrast_p10",
                f"{radius_name}_contrast_median",
                f"{radius_name}_contrast_frac_gt25",
                f"{radius_name}_contrast_frac_gt75",
            ]
        )
    names.extend(
        [
            "outer_axis_core_min",
            "outer_contrast_core_min",
            "outer_support_frac_150_25",
            "outer_support_frac_200_50",
        ]
    )
    return names


def summarize_arm_tensor(
    tensor: np.ndarray,
    *,
    ring_n: int,
    radii_mm: Sequence[float],
    hu_clip: Sequence[float],
) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=float)
    expected_channels = 1 + ring_n * len(radii_mm)
    if tensor.ndim != 2 or tensor.shape[1] != expected_channels:
        raise ValueError(
            f"arm tensor must have shape (profile_n, {expected_channels})"
        )
    axis = clip_hu(tensor[:, 0], hu_clip).astype(float)
    features = [
        axis.mean(),
        axis.std(),
        axis.min(),
        np.percentile(axis, 10),
        np.median(axis),
        np.percentile(axis, 90),
        axis.max(),
        np.mean(axis > 150),
        np.mean(axis > 250),
        np.mean(np.abs(np.diff(axis))),
    ]
    outer_contrast = None
    offset = 1
    for _ in radii_mm:
        ring = clip_hu(
            tensor[:, offset : offset + ring_n].mean(axis=1), hu_clip
        ).astype(float)
        contrast = axis - ring
        features.extend(
            [
                ring.mean(),
                ring.std(),
                contrast.mean(),
                contrast.min(),
                np.percentile(contrast, 10),
                np.median(contrast),
                np.mean(contrast > 25),
                np.mean(contrast > 75),
            ]
        )
        outer_contrast = contrast
        offset += ring_n
    if outer_contrast is None:
        raise ValueError("at least one radius is required")
    core = slice(3, -3) if len(axis) > 6 else slice(None)
    features.extend(
        [
            np.min(axis[core]),
            np.min(outer_contrast[core]),
            np.mean((axis > 150) & (outer_contrast > 25)),
            np.mean((axis > 200) & (outer_contrast > 50)),
        ]
    )
    result = np.asarray(features, dtype=np.float32)
    if len(result) != len(arm_feature_names(radii_mm)):
        raise RuntimeError("arm feature count mismatch")
    return result


def aggregate_arm_features(
    arm_features: np.ndarray,
    feature_names: Sequence[str],
    aggregations: Sequence[str],
) -> dict[str, float]:
    arm_features = np.asarray(arm_features, dtype=float)
    if arm_features.shape != (3, len(feature_names)):
        raise ValueError("arm_features must have shape (3, n_features)")
    functions = {
        "mean": lambda values: values.mean(axis=0),
        "min": lambda values: values.min(axis=0),
        "max": lambda values: values.max(axis=0),
        "std": lambda values: values.std(axis=0),
    }
    output = {}
    for aggregation in aggregations:
        if aggregation not in functions:
            raise ValueError(f"unsupported arm aggregation: {aggregation}")
        values = functions[aggregation](arm_features)
        output.update(
            {
                f"ct_{aggregation}_{name}": float(value)
                for name, value in zip(feature_names, values)
            }
        )
    return output


def expected_affine(row: pd.Series) -> np.ndarray:
    affine = np.eye(4, dtype=float)
    for axis in range(3):
        for column in range(4):
            affine[axis, column] = float(row[f"affine_{axis}{column}"])
    return affine


def validate_inputs(
    plan: pd.DataFrame,
    endpoints: pd.DataFrame,
    config: dict,
) -> None:
    if set(plan["split"].unique()) != set(config["splits"]):
        raise RuntimeError("candidate plan contains an unexpected split")
    if "test" in set(plan["split"]):
        raise RuntimeError("held-out test entered train/val CT extraction")
    if len(plan) != int(config["expected_candidate_rows"]):
        raise RuntimeError(f"unexpected candidate count: {len(plan)}")
    if not plan["candidate_id"].is_unique:
        raise RuntimeError("candidate_id is not unique")
    if set(plan["degree"]) != {3}:
        raise RuntimeError("CT baseline currently supports degree-3 only")
    endpoint_counts = endpoints.groupby("candidate_id").size()
    if set(endpoint_counts.index) != set(plan["candidate_id"]):
        raise RuntimeError("candidate IDs differ between plan and endpoint table")
    if not bool((endpoint_counts == 3).all()):
        raise RuntimeError("every candidate must contain exactly three endpoints")
    expected_counts = config["expected_patient_counts"]
    for split, count in expected_counts.items():
        actual_ids = set(
            plan.loc[plan["split"] == split, "scan_id"].astype(int).unique()
        )
        if len(actual_ids) != int(count):
            raise RuntimeError(f"unexpected {split} patient count: {len(actual_ids)}")
        if actual_ids != set(map(int, config["expected_patient_ids"][split])):
            raise RuntimeError(f"unexpected frozen patient IDs for {split}")
    missing = set(METADATA_COLUMNS + GEOMETRY_FEATURES) - set(plan.columns)
    if missing:
        raise RuntimeError(f"candidate plan is missing columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generator_config = json.loads(
        resolve_repo_path(config["generator_config"]).read_text(encoding="utf-8")
    )
    if generator_config.get("generator_status") != "frozen_train_val_v1":
        raise RuntimeError("JUNCTION generator is not frozen_train_val_v1")

    from extract_radial_features_z04 import (
        SplitZipReader,
        nifti_from_gz_bytes,
        sample_world,
    )
    from build_junction_plan import junction_geometry

    plan = pd.read_csv(resolve_repo_path(config["plan"]))
    endpoints = pd.read_csv(resolve_repo_path(config["endpoints"]))
    validate_inputs(plan, endpoints, config)
    endpoints_by_candidate = {
        candidate_id: group.sort_values("endpoint_index")
        for candidate_id, group in endpoints.groupby("candidate_id", sort=False)
    }

    expected_geometry = pd.read_csv(
        resolve_repo_path(config["expected_geometry"])
    ).set_index("scan_id")
    expected_alignment = pd.read_csv(
        resolve_repo_path(config["expected_alignment"])
    ).set_index("scan_id")
    archive_dir = resolve_repo_path(config["archive_dir"])
    reader = SplitZipReader(archive_dir / "801-1000.z04", archive_dir / "801-1000.zip")

    profile_n = int(config["profile_n"])
    ring_n = int(config["ring_n"])
    radii_mm = tuple(float(value) for value in config["radii_mm"])
    channels = 1 + ring_n * len(radii_mm)
    feature_names = arm_feature_names(radii_mm)
    candidate_count = len(plan)
    tensors = np.empty((candidate_count, 3, profile_n, channels), dtype=np.float32)
    arm_features = np.empty(
        (candidate_count, 3, len(feature_names)), dtype=np.float32
    )
    centers_lps = np.empty((candidate_count, 3), dtype=np.float64)
    endpoints_lps = np.empty((candidate_count, 3, 3), dtype=np.float64)
    tangents_lps = np.empty((candidate_count, 3, 3), dtype=np.float64)
    feature_rows: list[dict[str, object] | None] = [None] * candidate_count
    alignment_rows = []

    plan = plan.reset_index(drop=True)
    for patient_index, scan_id in enumerate(sorted(plan["scan_id"].unique()), 1):
        patient_rows = plan.index[plan["scan_id"] == scan_id].tolist()
        split_values = set(plan.loc[patient_rows, "split"])
        if len(split_values) != 1:
            raise RuntimeError(f"patient {scan_id} appears in multiple splits")
        split = next(iter(split_values))
        if int(scan_id) not in expected_geometry.index:
            raise RuntimeError(f"missing expected CT geometry for {scan_id}")
        if int(scan_id) not in expected_alignment.index:
            raise RuntimeError(f"missing expected CT alignment for {scan_id}")
        expected = expected_geometry.loc[int(scan_id)]
        expected_ct = expected_alignment.loc[int(scan_id)]
        if str(expected["split"]) != split or str(expected_ct["split"]) != split:
            raise RuntimeError(f"split mismatch for patient {scan_id}")

        archive_name = f"801-1000/{int(scan_id)}.img.nii.gz"
        raw, _ = reader.extract_entry_bytes(archive_name)
        ct_sha256 = hashlib.sha256(raw).hexdigest()
        meta, ct, uncompressed_blob = nifti_from_gz_bytes(raw)
        expected_shape = tuple(int(expected[f"shape_{axis}"]) for axis in "xyz")
        expected_spacing = tuple(
            float(expected[f"spacing_{axis}"]) for axis in "xyz"
        )
        shape_ok = tuple(meta["shape"]) == expected_shape
        spacing_ok = bool(
            np.allclose(meta["spacing"], expected_spacing, atol=1e-7, rtol=0)
        )
        affine_error = float(
            np.max(np.abs(meta["affine"] - expected_affine(expected)))
        )
        affine_ok = affine_error <= 1e-6
        sha_ok = ct_sha256 == str(expected_ct["ct_sha256"])
        if not (shape_ok and spacing_ok and affine_ok and sha_ok):
            raise RuntimeError(
                f"CT alignment failed for {scan_id}: shape={shape_ok}, "
                f"spacing={spacing_ok}, affine={affine_ok}, sha={sha_ok}"
            )
        alignment_rows.append(
            {
                "scan_id": int(scan_id),
                "split": split,
                "ct_sha256": ct_sha256,
                "mask_sha256": str(expected["mask_sha256"]),
                "shape_match": shape_ok,
                "spacing_match": spacing_ok,
                "affine_max_abs_error": affine_error,
                "affine_match": affine_ok,
                "sha256_match": sha_ok,
            }
        )
        inverse_affine = np.linalg.inv(meta["affine"])
        print(
            f"[{patient_index:02d}/{plan['scan_id'].nunique()}] {scan_id} "
            f"{split}: {len(patient_rows)} candidates"
        )
        for row_index in patient_rows:
            row = plan.loc[row_index]
            candidate_id = str(row["candidate_id"])
            candidate_endpoints = endpoints_by_candidate[candidate_id]
            xyz = candidate_endpoints[["x_lps", "y_lps", "z_lps"]].to_numpy(
                dtype=float
            )
            tangents = candidate_endpoints[
                [
                    "tangent_to_candidate_x",
                    "tangent_to_candidate_y",
                    "tangent_to_candidate_z",
                ]
            ].to_numpy(dtype=float)
            calculated_geometry = junction_geometry(xyz, tangents)
            if not all(
                math.isclose(
                    float(row[column]),
                    calculated_geometry[column],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for column in GEOMETRY_FEATURES
            ):
                raise RuntimeError(
                    f"geometry/endpoint mismatch for candidate {candidate_id}"
                )
            center, raw_tensor = sample_candidate_tensor(
                xyz,
                tangents,
                ct,
                inverse_affine,
                sample_world,
                profile_n=profile_n,
                ring_n=ring_n,
                radii_mm=radii_mm,
                longitudinal_range=config["longitudinal_range"],
            )
            per_arm = np.stack(
                [
                    summarize_arm_tensor(
                        arm_tensor,
                        ring_n=ring_n,
                        radii_mm=radii_mm,
                        hu_clip=config["hu_clip"],
                    )
                    for arm_tensor in raw_tensor
                ]
            )
            tensor = clip_hu(raw_tensor, config["hu_clip"])
            tensors[row_index] = tensor
            arm_features[row_index] = per_arm
            centers_lps[row_index] = center
            endpoints_lps[row_index] = xyz
            tangents_lps[row_index] = tangents
            output = {column: row[column] for column in METADATA_COLUMNS}
            output.update({column: float(row[column]) for column in GEOMETRY_FEATURES})
            output.update(
                aggregate_arm_features(
                    per_arm,
                    feature_names,
                    config["arm_aggregations"],
                )
            )
            feature_rows[row_index] = output
        del ct, raw, uncompressed_blob

    if any(row is None for row in feature_rows):
        raise RuntimeError("one or more candidates were not processed")
    features = pd.DataFrame(feature_rows)
    ct_columns = [column for column in features if column.startswith("ct_")]
    if len(ct_columns) != len(feature_names) * len(config["arm_aggregations"]):
        raise RuntimeError(f"unexpected CT feature count: {len(ct_columns)}")
    numeric = features[list(GEOMETRY_FEATURES) + ct_columns].to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()):
        raise RuntimeError("non-finite model feature detected")

    out_dir = resolve_repo_path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "junction_relation_features_train_val.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(
        out_dir / "junction_ct_alignment_train_val.csv", index=False
    )
    np.savez_compressed(
        out_dir / "junction_ct_tensors_train_val.npz",
        candidate_id=np.asarray(
            plan["candidate_id"].astype(str).tolist(), dtype=np.str_
        ),
        hu=tensors,
        arm_features=arm_features,
        arm_feature_names=np.asarray(feature_names),
        centers_lps=centers_lps,
        endpoints_lps=endpoints_lps,
        tangents_to_candidate_lps=tangents_lps,
        profile_t=np.linspace(
            float(config["longitudinal_range"][0]),
            float(config["longitudinal_range"][1]),
            profile_n,
        ),
        radii_mm=np.asarray(radii_mm),
        hu_clip=np.asarray(config["hu_clip"], dtype=float),
        geometry_feature_names=np.asarray(GEOMETRY_FEATURES),
        arm_aggregations=np.asarray(config["arm_aggregations"]),
        feature_version=np.asarray(config["feature_version"]),
    )
    print("DONE")
    print(f"patients: {len(alignment_rows)}")
    print(f"candidates: {len(features)}")
    print(f"geometry features: {len(GEOMETRY_FEATURES)}")
    print(f"CT features: {len(ct_columns)}")
    print(f"HU tensor: {tensors.shape}")
    print(f"arm summaries: {arm_features.shape}")
    print(f"output: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()