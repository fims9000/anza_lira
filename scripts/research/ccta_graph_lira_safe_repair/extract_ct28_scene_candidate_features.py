#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

import build_ct28_scene_plan as scene_builder
from ct28_perturbations import perturb_scene_endpoints
from extract_junction_ct_features import (
    aggregate_arm_features,
    arm_feature_names,
    expected_affine,
    sample_candidate_tensor,
    summarize_arm_tensor,
)
from extract_radial_features_z04 import (
    RADII_MM,
    RING_N,
    PROFILE_N,
    SplitZipReader,
    nifti_from_gz_bytes,
    radial_features,
    sample_world,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_DIR = REPO_ROOT / "artifacts/varvara/ct28_graph_lira"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "varvara/anza_lira_local_data/imagecas_801_1000"
EXPECTED_GEOMETRY = REPO_ROOT / "artifacts/varvara/ct28_pair/expected_ct_geometry.csv"
EXPECTED_ALIGNMENT = REPO_ROOT / "artifacts/varvara/ct28_pair/ct_alignment.csv"
EXPECTED_PATIENT_IDS = {
    "train": {953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983},
    "val": {957, 961, 965, 966, 974},
}
PAIR_GEOMETRY_FEATURES = tuple(f"pair_g{index}" for index in range(7))
JUNCTION_GEOMETRY_FEATURES = (
    "junction_span_mm",
    "junction_residual_mean_mm",
    "junction_residual_max_mm",
    "junction_forward_min_mm",
    "junction_forward_mean_mm",
    "junction_alignment_min",
    "junction_alignment_mean",
    "junction_center_distance_max_mm",
    "junction_center_distance_std_mm",
)
JUNCTION_LONGITUDINAL_RANGE = (0.05, 0.95)
HU_CLIP = (-300.0, 1200.0)
ARM_AGGREGATIONS = ("mean", "min", "max", "std")


class SceneFeatureError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract verified radial CCTA features for all CT28 scene candidates."
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--base-position-noise-mm", type=float, default=0.0)
    parser.add_argument("--base-tangent-noise-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def pair_sampling_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    vector = second - first
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise SceneFeatureError("PAIR endpoints coincide")
    direction = vector / distance
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(direction[2]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    first_axis = np.cross(direction, reference)
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(direction, first_axis)
    second_axis /= np.linalg.norm(second_axis)
    centers = first[None, :] + np.linspace(0.05, 0.95, PROFILE_N)[:, None] * vector
    points = [centers]
    angles = np.linspace(0.0, 2.0 * np.pi, RING_N, endpoint=False)
    for radius in RADII_MM:
        points.extend(
            centers
            + radius
            * (np.cos(angle) * first_axis + np.sin(angle) * second_axis)[None, :]
            for angle in angles
        )
    return np.concatenate(points, axis=0)


def validate_world_bounds(
    points_lps: np.ndarray, inverse_affine: np.ndarray, shape: Sequence[int]
) -> None:
    points_ras = np.asarray(points_lps, dtype=float).copy()
    points_ras[..., :2] *= -1.0
    flat = points_ras.reshape(-1, 3)
    voxels = (
        inverse_affine @ np.c_[flat, np.ones(len(flat), dtype=float)].T
    ).T[:, :3]
    upper = np.asarray(shape, dtype=float) - 1.0
    if not bool(((voxels >= 0.0) & (voxels <= upper)).all()):
        raise SceneFeatureError("candidate sampling grid leaves the CT volume")


def validate_inputs(
    plans: pd.DataFrame, endpoints: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    if len(plans) != 1404 or len(endpoints) != 7020 or len(candidates) != 15092:
        raise SceneFeatureError("unexpected frozen scene row counts")
    if set(plans["split"]) != {"train", "val"}:
        raise SceneFeatureError("scene pack must contain train/validation only")
    for split, expected in EXPECTED_PATIENT_IDS.items():
        actual = set(plans.loc[plans["split"] == split, "scan_id"].astype(int))
        if actual != expected:
            raise SceneFeatureError(f"unexpected frozen patient IDs for {split}")
    if candidates[["scene_id", "candidate_id"]].duplicated().any():
        raise SceneFeatureError("duplicate scene/candidate key")
    endpoint_lookup = endpoints.set_index(["scene_id", "endpoint_id"])
    if not endpoint_lookup.index.is_unique:
        raise SceneFeatureError("duplicate scene/endpoint key")
    expected_degree = {"PAIR": 2, "JUNCTION": 3}
    for row in candidates.itertuples(index=False):
        endpoint_ids = str(row.endpoint_ids).split("|")
        if len(endpoint_ids) != expected_degree[str(row.candidate_type)]:
            raise SceneFeatureError(f"candidate degree mismatch: {row.candidate_id}")
        if any((row.scene_id, endpoint_id) not in endpoint_lookup.index for endpoint_id in endpoint_ids):
            raise SceneFeatureError(f"candidate references missing endpoint: {row.candidate_id}")
    merged = candidates.merge(
        plans[["scene_id", "scan_id", "split"]],
        on="scene_id",
        how="left",
        validate="many_to_one",
    )
    if merged[["scan_id", "split"]].isna().any().any():
        raise SceneFeatureError("candidate lacks scene metadata")
    return merged


def extract_pair_features(
    endpoint_rows: pd.DataFrame, ct: np.ndarray, inverse_affine: np.ndarray
) -> np.ndarray:
    xyz = endpoint_rows[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
    validate_world_bounds(pair_sampling_points(xyz[0], xyz[1]), inverse_affine, ct.shape)
    row = SimpleNamespace(
        **{
            f"p{point}_{axis}": xyz[point - 1, axis]
            for point in (1, 2)
            for axis in range(3)
        }
    )
    values = radial_features(row, ct, inverse_affine)
    if values.shape != (38,):
        raise SceneFeatureError(f"unexpected PAIR CT feature shape: {values.shape}")
    return values


def extract_junction_features(
    endpoint_rows: pd.DataFrame,
    ct: np.ndarray,
    inverse_affine: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    xyz = endpoint_rows[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
    tangents = endpoint_rows[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(
        dtype=float
    )
    _, tensor = sample_candidate_tensor(
        xyz,
        tangents,
        ct,
        inverse_affine,
        sample_world,
        profile_n=PROFILE_N,
        ring_n=RING_N,
        radii_mm=RADII_MM,
        longitudinal_range=JUNCTION_LONGITUDINAL_RANGE,
    )
    arm_features = np.stack(
        [
            summarize_arm_tensor(
                arm,
                ring_n=RING_N,
                radii_mm=RADII_MM,
                hu_clip=HU_CLIP,
            )
            for arm in tensor
        ]
    )
    aggregated = aggregate_arm_features(
        arm_features, feature_names, ARM_AGGREGATIONS
    )
    return np.asarray(list(aggregated.values()), dtype=float)


def endpoint_item(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "position": row[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float),
        "tangent": row[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(
            dtype=float
        ),
    }


def apply_base_stress(
    endpoints: pd.DataFrame,
    *,
    seed: int,
    position_noise_mm: float,
    tangent_noise_deg: float,
) -> pd.DataFrame:
    frames = [
        perturb_scene_endpoints(
            rows,
            seed=seed,
            repetition=-1,
            position_noise_mm=position_noise_mm,
            tangent_noise_deg=tangent_noise_deg,
        )
        for _, rows in endpoints.groupby("scene_id", sort=False)
    ]
    stressed = pd.concat(frames, ignore_index=True)
    if set(map(tuple, stressed[["scene_id", "endpoint_id"]].to_numpy())) != set(
        map(tuple, endpoints[["scene_id", "endpoint_id"]].to_numpy())
    ):
        raise SceneFeatureError("base stress changed endpoint identities")
    tangent_norms = np.linalg.norm(
        stressed[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(dtype=float),
        axis=1,
    )
    if not np.allclose(tangent_norms, 1.0, atol=1e-9, rtol=0):
        raise SceneFeatureError("base stress produced non-unit tangents")
    return stressed


def recompute_candidate_geometry(
    candidates: pd.DataFrame, endpoints: pd.DataFrame
) -> pd.DataFrame:
    output = candidates.copy()
    endpoint_lookup = endpoints.set_index(["scene_id", "endpoint_id"])
    for row_index, row in output.iterrows():
        endpoint_ids = str(row["endpoint_ids"]).split("|")
        endpoint_rows = endpoint_lookup.loc[
            [(str(row["scene_id"]), endpoint_id) for endpoint_id in endpoint_ids]
        ].reset_index()
        items = [endpoint_item(item) for _, item in endpoint_rows.iterrows()]
        if row["candidate_type"] == "PAIR":
            values = scene_builder.pair_geometry(*items)
            columns = PAIR_GEOMETRY_FEATURES
        else:
            values = scene_builder.oriented_junction_geometry(items)
            columns = JUNCTION_GEOMETRY_FEATURES
        output.loc[row_index, list(columns)] = values
    geometry_columns = list(PAIR_GEOMETRY_FEATURES + JUNCTION_GEOMETRY_FEATURES)
    pair_geometry = output.loc[
        output["candidate_type"].eq("PAIR"), list(PAIR_GEOMETRY_FEATURES)
    ].to_numpy(dtype=float)
    if not np.isfinite(pair_geometry).all():
        raise SceneFeatureError("non-finite stressed PAIR geometry")
    if not np.isfinite(
        output.loc[
            output["candidate_type"].eq("JUNCTION"),
            list(JUNCTION_GEOMETRY_FEATURES),
        ].to_numpy(dtype=float)
    ).all():
        raise SceneFeatureError("non-finite stressed JUNCTION geometry")
    if output[geometry_columns].isna().all(axis=1).any():
        raise SceneFeatureError("stressed candidate lacks geometry features")
    return output


def main() -> None:
    args = parse_args()
    stress = (args.base_position_noise_mm, args.base_tangent_noise_deg)
    if stress not in {(0.0, 0.0), (1.0, 30.0)}:
        raise SceneFeatureError(
            "base stress must be clean (0 mm, 0 deg) or frozen historical-style "
            "(1 mm, 30 deg)"
        )
    stressed_pack = stress == (1.0, 30.0)
    if stressed_pack and args.out_dir.resolve() == args.scene_dir.resolve():
        raise SceneFeatureError("stressed feature extraction requires a separate --out-dir")
    scene_plan_path = args.scene_dir / "scene_plan_train_val.csv"
    endpoint_path = args.scene_dir / "scene_endpoints_train_val.csv"
    candidate_path = args.scene_dir / "scene_candidates_train_val.csv"
    plans = pd.read_csv(scene_plan_path)
    endpoints = pd.read_csv(endpoint_path)
    candidates = pd.read_csv(candidate_path)
    validate_inputs(plans, endpoints, candidates)
    if stressed_pack:
        endpoints = apply_base_stress(
            endpoints,
            seed=args.seed,
            position_noise_mm=args.base_position_noise_mm,
            tangent_noise_deg=args.base_tangent_noise_deg,
        )
        candidates = recompute_candidate_geometry(candidates, endpoints)
    frame = validate_inputs(plans, endpoints, candidates).reset_index(drop=True)
    endpoint_lookup = endpoints.set_index(["scene_id", "endpoint_id"])
    if not endpoint_lookup.index.is_unique:
        raise SceneFeatureError("duplicate scene/endpoint key")

    expected_geometry = pd.read_csv(EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(EXPECTED_ALIGNMENT).set_index("scan_id")
    reader = SplitZipReader(
        args.archive_dir / "801-1000.z04", args.archive_dir / "801-1000.zip"
    )
    junction_arm_names = arm_feature_names(RADII_MM)
    pair_ct_columns = [f"pair_ct{index}" for index in range(38)]
    junction_ct_columns = [
        f"junction_{aggregation}_{name}"
        for aggregation in ARM_AGGREGATIONS
        for name in junction_arm_names
    ]
    if len(junction_ct_columns) != 152:
        raise SceneFeatureError("unexpected JUNCTION CT feature count")
    pair_values = np.full((len(frame), 38), np.nan, dtype=float)
    junction_values = np.full((len(frame), 152), np.nan, dtype=float)
    sampling_valid = np.ones(len(frame), dtype=bool)
    sampling_errors = np.full(len(frame), "", dtype=object)
    alignment_rows: list[dict[str, object]] = []

    patients = sorted(frame["scan_id"].astype(int).unique())
    for patient_index, scan_id in enumerate(patients, 1):
        indices = frame.index[frame["scan_id"].astype(int) == scan_id].tolist()
        split_values = set(frame.loc[indices, "split"])
        if len(split_values) != 1:
            raise SceneFeatureError(f"patient {scan_id} appears in multiple splits")
        split = str(next(iter(split_values)))
        if scan_id not in expected_geometry.index or scan_id not in expected_alignment.index:
            raise SceneFeatureError(f"missing frozen alignment source for {scan_id}")
        expected = expected_geometry.loc[scan_id]
        expected_ct = expected_alignment.loc[scan_id]
        if str(expected["split"]) != split or str(expected_ct["split"]) != split:
            raise SceneFeatureError(f"split mismatch for patient {scan_id}")

        raw, _ = reader.extract_entry_bytes(f"801-1000/{scan_id}.img.nii.gz")
        ct_sha256 = hashlib.sha256(raw).hexdigest()
        metadata, ct, uncompressed_blob = nifti_from_gz_bytes(raw)
        expected_shape = tuple(int(expected[f"shape_{axis}"]) for axis in "xyz")
        expected_spacing = tuple(float(expected[f"spacing_{axis}"]) for axis in "xyz")
        shape_ok = tuple(metadata["shape"]) == expected_shape
        spacing_ok = bool(
            np.allclose(metadata["spacing"], expected_spacing, atol=1e-7, rtol=0)
        )
        affine_error = float(
            np.max(np.abs(metadata["affine"] - expected_affine(expected)))
        )
        affine_ok = affine_error <= 1e-6
        sha_ok = ct_sha256 == str(expected_ct["ct_sha256"])
        if not (shape_ok and spacing_ok and affine_ok and sha_ok):
            raise SceneFeatureError(
                f"CT alignment failed for {scan_id}: shape={shape_ok}, "
                f"spacing={spacing_ok}, affine={affine_ok}, sha={sha_ok}"
            )
        alignment_rows.append(
            {
                "scan_id": scan_id,
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
        inverse_affine = np.linalg.inv(metadata["affine"])
        print(
            f"[{patient_index:02d}/{len(patients)}] {scan_id} {split}: "
            f"{len(indices)} candidates",
            flush=True,
        )
        for row_index in indices:
            row = frame.loc[row_index]
            endpoint_ids = str(row["endpoint_ids"]).split("|")
            endpoint_rows = endpoint_lookup.loc[
                [(str(row["scene_id"]), endpoint_id) for endpoint_id in endpoint_ids]
            ].reset_index()
            try:
                if row["candidate_type"] == "PAIR":
                    pair_values[row_index] = extract_pair_features(
                        endpoint_rows, ct, inverse_affine
                    )
                else:
                    junction_values[row_index] = extract_junction_features(
                        endpoint_rows, ct, inverse_affine, junction_arm_names
                    )
            except (SceneFeatureError, ValueError) as error:
                if "leaves the CT volume" not in str(error):
                    raise
                sampling_valid[row_index] = False
                sampling_errors[row_index] = "sampling_grid_outside_ct"
        del ct, raw, uncompressed_blob

    pair_mask = frame["candidate_type"].eq("PAIR").to_numpy()
    junction_mask = ~pair_mask
    pair_finite = np.isfinite(pair_values[pair_mask])
    junction_finite = np.isfinite(junction_values[junction_mask])
    if not np.all(pair_finite.all(axis=1) | (~pair_finite).all(axis=1)):
        raise SceneFeatureError("partially missing PAIR CT feature vector")
    if not np.all(junction_finite.all(axis=1) | (~junction_finite).all(axis=1)):
        raise SceneFeatureError("partially missing JUNCTION CT feature vector")
    valid_from_features = np.zeros(len(frame), dtype=bool)
    valid_from_features[pair_mask] = pair_finite.all(axis=1)
    valid_from_features[junction_mask] = junction_finite.all(axis=1)
    if not np.array_equal(valid_from_features, sampling_valid):
        raise SceneFeatureError("CT sampling audit does not match feature availability")

    output = pd.concat(
        [
            frame.reset_index(drop=True),
            pd.DataFrame(
                {
                    "ct_sampling_valid": sampling_valid.astype(int),
                    "ct_sampling_error": sampling_errors,
                }
            ),
            pd.DataFrame(pair_values, columns=pair_ct_columns),
            pd.DataFrame(junction_values, columns=junction_ct_columns),
        ],
        axis=1,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if stressed_pack:
        derived_plan_path = args.out_dir / "scene_plan_train_val.csv"
        derived_endpoint_path = args.out_dir / "scene_endpoints_train_val.csv"
        derived_candidate_path = args.out_dir / "scene_candidates_train_val.csv"
        shutil.copyfile(scene_plan_path, derived_plan_path)
        endpoints.to_csv(derived_endpoint_path, index=False)
        candidates.to_csv(derived_candidate_path, index=False)
    else:
        derived_plan_path = scene_plan_path
        derived_endpoint_path = endpoint_path
        derived_candidate_path = candidate_path
    feature_path = args.out_dir / "scene_candidate_features_train_val.csv"
    alignment_path = args.out_dir / "scene_ct_alignment_train_val.csv"
    protocol_path = args.out_dir / "scene_candidate_feature_protocol.json"
    output.to_csv(feature_path, index=False)
    pd.DataFrame(alignment_rows).to_csv(alignment_path, index=False)
    protocol = {
        "protocol_version": (
            "ct28_scene_candidate_radial_ct_stress30_jitter1_v1"
            if stressed_pack
            else "ct28_scene_candidate_radial_ct_v1"
        ),
        "held_out_test_accessed": False,
        "candidate_rows": len(output),
        "patient_counts": {split: len(ids) for split, ids in EXPECTED_PATIENT_IDS.items()},
        "alignment_checks_passed": len(alignment_rows),
        "ct_sampling_valid_candidates": int(sampling_valid.sum()),
        "ct_sampling_unavailable_candidates": int((~sampling_valid).sum()),
        "ct_sampling_unavailable_by_type": {
            candidate_type: int(
                ((frame["candidate_type"] == candidate_type).to_numpy() & ~sampling_valid).sum()
            )
            for candidate_type in ("PAIR", "JUNCTION")
        },
        "ct_sampling_unavailable_policy": (
            "Keep the candidate and store NaN CT features; the local model applies "
            "train-fitted median imputation without missing indicators. Never sample "
            "beyond the CT boundary."
        ),
        "pair_ct_feature_count": len(pair_ct_columns),
        "junction_ct_feature_count": len(junction_ct_columns),
        "pair_representation": {
            "profile_n": PROFILE_N,
            "ring_n": RING_N,
            "radii_mm": list(RADII_MM),
            "hu_clip": list(HU_CLIP),
            "feature_columns": pair_ct_columns,
        },
        "junction_representation": {
            "candidate_tensor_shape": [3, PROFILE_N, 1 + RING_N * len(RADII_MM)],
            "longitudinal_range": list(JUNCTION_LONGITUDINAL_RANGE),
            "arm_summary_features": junction_arm_names,
            "arm_aggregations": list(ARM_AGGREGATIONS),
            "feature_columns": junction_ct_columns,
        },
        "coordinate_conversion": "ImageCAS-X VTK LPS to ImageCAS NIfTI RAS by negating x,y before inverse affine",
        "base_stress": {
            "enabled": stressed_pack,
            "position_noise_mm_sigma": args.base_position_noise_mm,
            "tangent_noise_deg_sigma": args.base_tangent_noise_deg,
            "seed": args.seed,
            "seed_repetition": -1 if stressed_pack else None,
            "role": (
                "single deterministic stressed realization used for both local-model "
                "training and validation; not consistency augmentation"
                if stressed_pack
                else "clean controlled scene"
            ),
        },
        "source_files": {
            "scene_plan": provenance_path(scene_plan_path),
            "scene_endpoints": provenance_path(endpoint_path),
            "scene_candidates": provenance_path(candidate_path),
            "expected_geometry": provenance_path(EXPECTED_GEOMETRY),
            "expected_alignment": provenance_path(EXPECTED_ALIGNMENT),
        },
        "source_sha256": {
            "scene_plan": sha256(scene_plan_path),
            "scene_endpoints": sha256(endpoint_path),
            "scene_candidates": sha256(candidate_path),
            "expected_geometry": sha256(EXPECTED_GEOMETRY),
            "expected_alignment": sha256(EXPECTED_ALIGNMENT),
        },
        "output_sha256": {
            "scene_candidate_features_train_val.csv": sha256(feature_path),
            "scene_ct_alignment_train_val.csv": sha256(alignment_path),
            "scene_plan_train_val.csv": sha256(derived_plan_path),
            "scene_endpoints_train_val.csv": sha256(derived_endpoint_path),
            "scene_candidates_train_val.csv": sha256(derived_candidate_path),
        },
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print("DONE", flush=True)
    print(f"patients: {len(alignment_rows)}", flush=True)
    print(f"candidates: {len(output)}", flush=True)
    print(f"PAIR CT features: {len(pair_ct_columns)}", flush=True)
    print(f"JUNCTION CT features: {len(junction_ct_columns)}", flush=True)
    print(f"output: {provenance_path(feature_path)}", flush=True)


if __name__ == "__main__":
    main()
