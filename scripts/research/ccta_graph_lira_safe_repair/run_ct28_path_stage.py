#!/usr/bin/env python3
"""Train-only path parameter selection and validation-only CT28 path evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import ct28_path_stage as path_stage
import run_ct28_graph_lira as graph
from extract_radial_features_z04 import nifti_from_gz_bytes


ROOT = graph.REPO_ROOT
SCENES = graph.DEFAULT_SCENE_DIR
SOURCE = ROOT / "varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/clean_eval"
SEGMENTATIONS = ROOT / "varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations"
JUNCTION_TRUTH = ROOT / "varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/metadata/junctions_truth.csv"
OUTPUT = ROOT / "varvara/anza_lira_local_results/ct28_path_protocol_v1"
TEST_IDS = {954, 958, 972, 973, 980, 984}
TRAIN_IDS = {953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983}
VAL_IDS = {957, 961, 965, 966, 974}
PARAMETER_GRID = tuple(
    path_stage.PathConfig(radius, center, scale)
    for radius in (1.5, 2.0, 2.5)
    for center in (1.0, 2.0)
    for scale in (100.0, 200.0)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "evaluate", "both"), default="both")
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--train-scenes-per-type-patient", type=int, default=1)
    parser.add_argument("--chosen-config", type=Path,
                        help="Train-selected path config; defaults to out-dir/chosen_config_train.json")
    parser.add_argument("--slack-fraction", type=float, default=0.0,
                        help="Diagnostic near-optimal bottleneck slack; zero retains v1 max-min routing")
    parser.add_argument("--ignore-unavoidable-endpoint-contact", action="store_true",
                        help="Count a decoy contact only away from a selected endpoint's 0.75 mm neighborhood")
    parser.add_argument("--avoid-other-endpoints", action="store_true",
                        help="Exclude voxels near unselected scene endpoints except at selected endpoint starts")
    return parser.parse_args()


def load_scene_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plans = pd.read_csv(SCENES / "scene_plan_train_val.csv")
    endpoints = pd.read_csv(SCENES / "scene_endpoints_train_val.csv")
    candidates = pd.read_csv(SCENES / "scene_candidates_train_val.csv")
    if set(plans.scan_id) != TRAIN_IDS | VAL_IDS or set(plans.split) != {"train", "val"}:
        raise ValueError("frozen train/validation cohort changed")
    if set(plans.scan_id) & TEST_IDS or set(candidates.scene_id) - set(plans.scene_id):
        raise ValueError("held-out or unknown scene entered path experiment")
    return plans, endpoints, candidates


def load_segmentation(patient: int, shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    metadata, segmentation, _ = nifti_from_gz_bytes(
        (SEGMENTATIONS / f"{patient}.coronary.nii.gz").read_bytes()
    )
    if tuple(metadata["shape"]) != shape or not np.allclose(metadata["affine"], affine, atol=1e-6, rtol=0):
        raise ValueError(f"segmentation geometry differs from verified CT: {patient}")
    return np.asarray(segmentation) > 0


def mask_tree(segmentation: np.ndarray, affine: np.ndarray) -> cKDTree:
    voxels = np.argwhere(segmentation)
    if len(voxels) == 0:
        raise ValueError("empty coronary segmentation")
    return cKDTree(path_stage.voxel_to_lps(voxels, affine))


def selected_components(scene_id: str, candidate_ids: list[str],
                        candidates: pd.DataFrame, endpoints: pd.DataFrame) -> tuple[list[tuple[str, np.ndarray, np.ndarray, set[str]]], dict[str, np.ndarray]]:
    scene_candidates = candidates[candidates.scene_id.eq(scene_id)].set_index("candidate_id")
    endpoint_lookup = endpoints[endpoints.scene_id.eq(scene_id)].set_index("endpoint_id")
    all_points = {str(key): row[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
                  for key, row in endpoint_lookup.iterrows()}
    components = []
    for candidate_id in candidate_ids:
        candidate = scene_candidates.loc[candidate_id]
        ids = str(candidate.endpoint_ids).split("|")
        rows = endpoint_lookup.loc[ids]
        if len(ids) not in (2, 3) or len(set(ids)) != len(ids):
            raise ValueError("unsupported or repeated path endpoint")
        components.append((str(candidate.candidate_type),
                           rows[["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float),
                           rows[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(dtype=float),
                           set(ids)))
    return components, all_points


def validate_route_geometry(result: path_stage.PathResult, positions: np.ndarray,
                            affine: np.ndarray) -> None:
    if result.status != "ok":
        return
    snapped = np.rint(path_stage.lps_to_voxel(positions, np.linalg.inv(affine))).astype(int)
    expected_paths = 1 if len(positions) == 2 else len(positions)
    if len(result.paths_voxel) != expected_paths:
        raise ValueError("path component count differs from endpoint count")
    for path, start in zip(result.paths_voxel, snapped):
        voxels = np.asarray(path, dtype=int)
        if len(voxels) < 2 or not np.array_equal(voxels[0], start):
            raise ValueError("path has no segment or starts at the wrong endpoint")
        if (len(set(path)) != len(path) or
                np.any(np.max(np.abs(np.diff(voxels, axis=0)), axis=1) != 1)):
            raise ValueError("path repeats a voxel or has a non-26-neighbor step")
    if len(positions) == 2:
        if not np.array_equal(np.asarray(result.paths_voxel[0][-1]), snapped[1]):
            raise ValueError("PAIR path ends at the wrong endpoint")
    elif len({path[-1] for path in result.paths_voxel}) != 1:
        raise ValueError("JUNCTION arms do not meet at one common voxel")


def evaluate_routes(
    ct: np.ndarray, affine: np.ndarray, tree: cKDTree,
    components: list[tuple[str, np.ndarray, np.ndarray, set[str]]],
    all_points: dict[str, np.ndarray], config: path_stage.PathConfig,
    solver=None,
    ignore_unavoidable_endpoint_contact: bool = False,
) -> tuple[dict, list[dict]]:
    routes = []
    route_records = []
    selected_ids = set().union(*(item[3] for item in components))
    other_points = np.asarray([point for key, point in all_points.items() if key not in selected_ids])
    total_lengths, all_world, mask_hits, bottlenecks = [], [], [], []
    for kind, positions, tangents, ids in components:
        try:
            result = (solver(ct, affine, positions, tangents, config, other_points)
                      if solver else path_stage.solve_path(ct, affine, positions, tangents, config))
        except ValueError as error:
            result = path_stage.PathResult(str(error), 0.0, 0.0, (), None,
                                           float("nan"), float("nan"), float("nan"))
        validate_route_geometry(result, positions, affine)
        routes.append(result)
        path_voxels = [voxel for arm in result.paths_voxel for voxel in arm]
        world = path_stage.voxel_to_lps(np.asarray(path_voxels), affine) if path_voxels else np.empty((0, 3))
        all_world.append(world)
        mask_fraction = float(np.mean(tree.query(world)[0] <= 0.75)) if len(world) else 0.0
        mask_hits.append(mask_fraction)
        bottlenecks.append(result.bottleneck)
        total_lengths.append(result.length_mm)
        decoy_close = (np.linalg.norm(world[:, None, :] - other_points[None, :, :], axis=2)
                       <= config.extra_endpoint_clearance_mm).any(axis=1) if len(world) and len(other_points) else np.zeros(len(world), dtype=bool)
        endpoint_close = (np.linalg.norm(world[:, None, :] - positions[None, :, :], axis=2)
                          <= config.extra_endpoint_clearance_mm).any(axis=1) if len(world) else np.zeros(0, dtype=bool)
        unavoidable_contact = bool(np.any(decoy_close & endpoint_close))
        extra_hit = bool(np.any(decoy_close & ~endpoint_close)) if ignore_unavoidable_endpoint_contact else bool(np.any(decoy_close))
        record = {
            "kind": kind, "status": result.status, "endpoint_ids": sorted(ids),
            "paths_voxel": result.paths_voxel, "center_lps": result.center_lps,
            "bottleneck": result.bottleneck, "length_mm": result.length_mm,
            "vessel_hu": result.vessel_hu if np.isfinite(result.vessel_hu) else None,
            "background_hu": result.background_hu if np.isfinite(result.background_hu) else None,
            "contrast_hu": result.contrast_hu if np.isfinite(result.contrast_hu) else None,
            "mask_fraction_within_0p75mm": mask_fraction,
            "extra_endpoint_hit": extra_hit,
        }
        if ignore_unavoidable_endpoint_contact:
            record["decoy_near_selected_endpoint"] = unavoidable_contact
        route_records.append(record)
    cross_touch = False
    if len(all_world) == 2 and all(len(points) for points in all_world):
        cross_touch = bool(np.min(cKDTree(all_world[0]).query(all_world[1])[0])
                           <= config.extra_endpoint_clearance_mm)
    path_success = all(route.status == "ok" for route in routes) and not any(
        row["extra_endpoint_hit"] for row in route_records
    ) and not cross_touch
    straight = sum(float(np.linalg.norm(item[1][0] - item[1][-1]))
                   if item[0] == "PAIR" else
                   sum(float(np.linalg.norm(point - np.mean(item[1], axis=0))) for point in item[1])
                   for item in components)
    metrics = {
        "path_success": bool(path_success),
        "route_status": "ok" if path_success else "invalid_or_extra_connection",
        "component_statuses": "|".join(route.status for route in routes),
        "extra_endpoint_hit": any(row["extra_endpoint_hit"] for row in route_records),
        "cross_relation_touch": cross_touch,
        "bottleneck": min(bottlenecks) if bottlenecks else 0.0,
        "total_length_mm": sum(total_lengths),
        "length_over_straight": sum(total_lengths) / max(straight, 1e-6),
        "mask_fraction_within_0p75mm": float(np.mean(mask_hits)),
    }
    if ignore_unavoidable_endpoint_contact:
        metrics["decoy_near_selected_endpoint"] = any(
            row["decoy_near_selected_endpoint"] for row in route_records)
    return metrics, route_records


def patient_inputs(patient: int, archive_dir: Path,
                   expected_geometry: pd.DataFrame, expected_alignment: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, cKDTree]:
    ct, inverse = graph.load_verified_ct(patient, archive_dir, expected_geometry, expected_alignment)
    affine = np.linalg.inv(inverse)
    segmentation = load_segmentation(patient, ct.shape[:3], affine)
    return ct, affine, mask_tree(segmentation, affine)


def tune(args: argparse.Namespace, plans: pd.DataFrame, endpoints: pd.DataFrame,
         candidates: pd.DataFrame, expected_geometry: pd.DataFrame,
         expected_alignment: pd.DataFrame) -> path_stage.PathConfig:
    if args.train_scenes_per_type_patient < 1:
        raise ValueError("train-scenes-per-type-patient must be positive")
    choices = plans[(plans.split.eq("train")) &
                    (plans.scene_family.isin(("pair_only", "junction_only")))].copy()
    choices = choices.sort_values("scene_id").groupby(
        ["scan_id", "scene_family"], sort=True
    ).head(args.train_scenes_per_type_patient)
    rows = []
    for patient in sorted(TRAIN_IDS):
        patient_plans = choices[choices.scan_id.eq(patient)]
        ct, affine, tree = patient_inputs(patient, args.archive_dir, expected_geometry, expected_alignment)
        for plan in patient_plans.itertuples():
            positive = candidates[(candidates.scene_id.eq(plan.scene_id)) & candidates.y.eq(1)]
            expected_kind = "PAIR" if plan.scene_family == "pair_only" else "JUNCTION"
            positive = positive[positive.candidate_type.eq(expected_kind)]
            if len(positive) != 1:
                raise ValueError("train path calibration needs exactly one true candidate")
            components, all_points = selected_components(
                plan.scene_id, [str(positive.iloc[0].candidate_id)], candidates, endpoints
            )
            for index, config in enumerate(PARAMETER_GRID):
                metrics, _ = evaluate_routes(ct, affine, tree, components, all_points, config)
                rows.append({"grid_index": index, "scene_id": plan.scene_id,
                             "scan_id": patient, "scene_family": plan.scene_family, **metrics})
        print(f"train path calibration: patient {patient} done", flush=True)
        del ct, tree
    frame = pd.DataFrame(rows)
    if set(frame.scan_id) != TRAIN_IDS or frame.scene_id.nunique() < 2 * len(TRAIN_IDS):
        raise ValueError("train path calibration does not cover PAIR and JUNCTION per patient")
    frame.to_csv(args.out_dir / "tuning_train.csv", index=False)
    summaries = []
    for index, group in frame.groupby("grid_index"):
        success = group.path_success.astype(bool)
        score = float((group.mask_fraction_within_0p75mm * success).mean()
                      - 0.5 * (~success).mean()
                      - 0.25 * group.extra_endpoint_hit.astype(bool).mean()
                      - 0.02 * np.maximum(group.length_over_straight - 1, 0).clip(0, 5).mean())
        summaries.append({"grid_index": index, **asdict(PARAMETER_GRID[index]),
                          "score": score, "path_success": int(success.sum()),
                          "scenes": len(group),
                          "mean_mask_fraction": float(group.mask_fraction_within_0p75mm.mean()),
                          "extra_endpoint_hits": int(group.extra_endpoint_hit.sum())})
    summary = pd.DataFrame(summaries).sort_values(
        ["score", "corridor_radius_mm", "hu_difference_scale", "center_refine_mm"],
        ascending=[False, True, True, True]
    )
    summary.to_csv(args.out_dir / "tuning_summary_train.csv", index=False)
    chosen = PARAMETER_GRID[int(summary.iloc[0].grid_index)]
    (args.out_dir / "chosen_config_train.json").write_text(json.dumps({
        "status": "train_selected_development", "held_out_test_accessed": False,
        "selection_rule": "maximize mean mask-within-0.75mm on successful paths minus 0.5 failure, 0.25 extra endpoint, 0.02 length-excess penalty; deterministic smaller-parameter tie break",
        "train_scenes_per_type_patient": args.train_scenes_per_type_patient,
        "train_scenes": int(frame.scene_id.nunique()),
        "selected_grid_index": int(summary.iloc[0].grid_index),
        "selected_config": asdict(chosen),
        "grid": [asdict(config) for config in PARAMETER_GRID],
    }, indent=2) + "\n")
    return chosen


def evaluate(args: argparse.Namespace, config: path_stage.PathConfig,
             plans: pd.DataFrame, endpoints: pd.DataFrame, candidates: pd.DataFrame,
             expected_geometry: pd.DataFrame, expected_alignment: pd.DataFrame) -> None:
    predictions_path = SOURCE / "graph_lira_predictions_val.csv"
    predictions = pd.read_csv(predictions_path)
    selected = predictions[predictions.model.eq("ct_conditioned")].copy()
    if len(selected) != 368 or set(selected.scan_id) != VAL_IDS or int(selected.accepted.sum()) != 52:
        raise ValueError("selected structural validation result differs from 52-scene reference")
    source_protocol = json.loads((SOURCE / "graph_lira_protocol.json").read_text())
    if source_protocol["source_sha256"]["scene_plan"] != graph.sha256(SCENES / "scene_plan_train_val.csv"):
        raise ValueError("structural predictions and frozen scenes differ")
    truth = pd.read_csv(JUNCTION_TRUTH)
    truth = truth[truth.split.eq("val")].set_index(["scan_id", "side", "point_id"])
    rows, route_lines = [], []
    for patient in sorted(VAL_IDS):
        patient_rows = selected[selected.scan_id.eq(patient)]
        ct, affine, tree = patient_inputs(patient, args.archive_dir, expected_geometry, expected_alignment)
        for row in patient_rows.itertuples():
            output = {"scene_id": row.scene_id, "scan_id": patient,
                      "scene_family": row.scene_family, "relation_class": row.relation_class,
                      "accepted": int(row.accepted), "structural_exact": int(row.exact),
                      "structural_false": int(row.false_scene),
                      "selected_relation": row.final_relation}
            if not row.accepted:
                rows.append({**output, "path_attempted": False, "path_success": False,
                             "route_status": "skipped_abstain_or_none"})
                continue
            ids = [candidate_id for candidate_id in (row.pair_candidate_id, row.junction_candidate_id)
                   if isinstance(candidate_id, str) and candidate_id]
            if not ids or row.final_relation == "NONE":
                raise ValueError("accepted repair lacks selected candidate")
            components, all_points = selected_components(row.scene_id, ids, candidates, endpoints)
            solver = lambda volume, image_affine, positions, tangents, settings, forbidden: (
                path_stage.solve_path_with_bottleneck_slack(
                    volume, image_affine, positions, tangents, settings, args.slack_fraction,
                    forbidden_lps=forbidden if args.avoid_other_endpoints else None))
            metrics, routes = evaluate_routes(ct, affine, tree, components, all_points,
                                              config, solver=solver,
                                              ignore_unavoidable_endpoint_contact=args.ignore_unavoidable_endpoint_contact)
            if len(routes) != len(ids):
                raise ValueError("path component count differs from selected relation")
            center_error = float("nan")
            if isinstance(row.junction_source_id, str) and any(route["kind"] == "JUNCTION" for route in routes):
                scan, side, point = row.junction_source_id.split(":")
                true_center = truth.loc[(int(scan), side, int(point))][["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
                actual_center = next((route["center_lps"] for route in routes if route["kind"] == "JUNCTION"), None)
                if actual_center is not None:
                    center_error = float(np.linalg.norm(np.asarray(actual_center) - true_center))
            rows.append({**output, "path_attempted": True, **metrics,
                         "junction_center_error_mm": center_error})
            route_lines.append({"scene_id": row.scene_id, "scan_id": patient,
                                "selected_candidate_ids": ids, "routes": routes})
        print(f"validation path evaluation: patient {patient} done", flush=True)
        del ct, tree
    frame = pd.DataFrame(rows)
    if len(frame) != 368 or int(frame.path_attempted.sum()) != 52:
        raise ValueError("path stage changed structural scene count or acceptance")
    frame.to_csv(args.out_dir / "path_metrics_val.csv", index=False)
    with (args.out_dir / "paths_val.jsonl").open("w") as handle:
        for item in route_lines:
            handle.write(json.dumps(item, allow_nan=False, default=lambda x: float(x) if isinstance(x, np.floating) else x) + "\n")
    attempted = frame[frame.path_attempted].copy()
    metrics = {
        "status": "validation_path_development", "held_out_test_accessed": False,
        "structural_accepted_unchanged": int(frame.accepted.sum()) == 52,
        "validation_scenes": 368, "path_attempted": len(attempted),
        "path_success": int(attempted.path_success.sum()),
        "accepted_structural_exact": int(attempted.structural_exact.sum()),
        "accepted_structural_false": int(attempted.structural_false.sum()),
        "extra_endpoint_hits": int(attempted.extra_endpoint_hit.sum()),
        "cross_relation_touches": int(attempted.cross_relation_touch.sum()),
        "path_length_over_straight_median": float(attempted.length_over_straight.median()),
        "path_length_over_straight_p90": float(attempted.length_over_straight.quantile(0.9)),
        "path_length_over_straight_max": float(attempted.length_over_straight.max()),
        "mask_fraction_within_0p75mm_mean": float(attempted.mask_fraction_within_0p75mm.mean()),
        "mask_fraction_within_0p75mm_min": float(attempted.mask_fraction_within_0p75mm.min()),
        "junction_center_error_mm_median": float(attempted.junction_center_error_mm.median()),
        "junction_center_error_mm_max": float(attempted.junction_center_error_mm.max()),
        "by_patient": attempted.groupby("scan_id").agg(
            attempted=("scene_id", "size"), success=("path_success", "sum")
        ).to_dict(orient="index"),
        "by_scene_family": attempted.groupby("scene_family").agg(
            attempted=("scene_id", "size"), success=("path_success", "sum"),
            mean_mask_fraction=("mask_fraction_within_0p75mm", "mean")
        ).to_dict(orient="index"),
        "source_predictions_sha256": graph.sha256(predictions_path),
        "source_scene_sha256": source_protocol["source_sha256"]["scene_plan"],
        "selected_config": asdict(config),
        "bottleneck_slack_fraction": args.slack_fraction,
        "ignore_unavoidable_endpoint_contact": args.ignore_unavoidable_endpoint_contact,
        "avoid_other_endpoints": args.avoid_other_endpoints,
    }
    (args.out_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.slack_fraction <= 0.2:
        raise ValueError("bottleneck slack must lie in [0, 0.2]")
    if args.phase in ("tune", "both") and args.slack_fraction:
        raise ValueError("slack ablation must be train-selected separately before validation")
    if args.avoid_other_endpoints and not args.ignore_unavoidable_endpoint_contact:
        raise ValueError("endpoint avoidance requires unavoidable-contact auditing")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plans, endpoints, candidates = load_scene_tables()
    expected_geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    expected_alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    if args.phase in ("tune", "both"):
        chosen = tune(args, plans, endpoints, candidates, expected_geometry, expected_alignment)
    else:
        saved = json.loads((args.chosen_config or args.out_dir / "chosen_config_train.json").read_text())
        if saved["status"] != "train_selected_development" or saved["held_out_test_accessed"]:
            raise ValueError("train-selected path config unavailable")
        chosen = path_stage.PathConfig(**saved["selected_config"])
    if args.phase in ("evaluate", "both"):
        evaluate(args, chosen, plans, endpoints, candidates, expected_geometry, expected_alignment)


if __name__ == "__main__":
    main()
