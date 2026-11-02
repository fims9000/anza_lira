#!/usr/bin/env python3
"""Post-hoc branch-specific audit of the 34 train scenes used for CT28 path tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_ct28_path_anatomy as anatomy
import run_ct28_graph_lira as graph
import run_ct28_path_stage as paths


ROOT = graph.REPO_ROOT
OUTPUT = ROOT / "varvara/anza_lira_local_results/ct28_pretest_audit"
CHOSEN = ROOT / "varvara/anza_lira_local_results/ct28_path_protocol_v1/chosen_config_train.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=graph.DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slack-fraction", type=float, default=0.0)
    parser.add_argument("--ignore-unavoidable-endpoint-contact", action="store_true")
    parser.add_argument("--avoid-other-endpoints", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.slack_fraction <= 0.2:
        raise ValueError("this train-only ablation is bounded to slack 0..0.2")
    if args.avoid_other_endpoints and not args.ignore_unavoidable_endpoint_contact:
        raise ValueError("endpoint avoidance requires unavoidable-contact auditing")
    plans, endpoints, candidates = paths.load_scene_tables()
    selected = json.loads(CHOSEN.read_text())
    if selected["train_scenes"] != 34 or selected["train_scenes_per_type_patient"] != 1:
        raise ValueError("path tuning scene rule changed")
    config = paths.path_stage.PathConfig(**selected["selected_config"])
    choices = plans[plans.split.eq("train") & plans.scene_family.isin(("pair_only", "junction_only"))]
    choices = choices.sort_values("scene_id").groupby(["scan_id", "scene_family"], sort=True).head(1)
    if len(choices) != 34 or set(choices.scan_id) != paths.TRAIN_IDS:
        raise ValueError("train path audit does not match tuning cohort")
    geometry = pd.read_csv(graph.ct_features.EXPECTED_GEOMETRY).set_index("scan_id")
    alignment = pd.read_csv(graph.ct_features.EXPECTED_ALIGNMENT).set_index("scan_id")
    positives = pd.read_csv(anatomy.JUNCTION / "junction_candidate_endpoints_train_val.csv")
    truth = pd.read_csv(anatomy.SOURCE / "metadata/junctions_truth.csv")
    truth = truth[truth.split.eq("train")].set_index(["scan_id", "side", "point_id"])
    cache, scene_rows, arm_rows = {}, [], []
    for patient in sorted(paths.TRAIN_IDS):
        ct, affine, mask_tree = paths.patient_inputs(patient, args.archive_dir, geometry, alignment)
        for scene in choices[choices.scan_id.eq(patient)].itertuples():
            kind = "PAIR" if scene.scene_family == "pair_only" else "JUNCTION"
            positive = candidates[candidates.scene_id.eq(scene.scene_id) & candidates.candidate_type.eq(kind) & candidates.y.eq(1)]
            if len(positive) != 1:
                raise ValueError("expected one true relation per train path scene")
            candidate_id = str(positive.iloc[0].candidate_id)
            components, all_points = paths.selected_components(scene.scene_id, [candidate_id], candidates, endpoints)
            solver = lambda volume, image_affine, positions, tangents, settings, forbidden: (
                paths.path_stage.solve_path_with_bottleneck_slack(
                    volume, image_affine, positions, tangents, settings, args.slack_fraction,
                    forbidden_lps=forbidden if args.avoid_other_endpoints else None))
            metrics, route_records = paths.evaluate_routes(
                ct, affine, mask_tree, components, all_points, config, solver=solver,
                ignore_unavoidable_endpoint_contact=args.ignore_unavoidable_endpoint_contact)
            if len(route_records) != 1:
                raise ValueError("unexpected train path component count")
            route = route_records[0]
            positions = components[0][1]
            if kind == "PAIR":
                references = [anatomy.pair_reference(patient, positions, cache)]
                center_error = float("nan")
            else:
                references = anatomy.junction_references(patient, str(scene.junction_source_id),
                                                         positions, positives, truth, cache)
                scan, side, point = str(scene.junction_source_id).split(":")
                true_center = truth.loc[(int(scan), side, int(point))][["x_lps", "y_lps", "z_lps"]].to_numpy(dtype=float)
                center_error = (float(np.linalg.norm(np.asarray(route["center_lps"]) - true_center))
                                if route["center_lps"] is not None else float("nan"))
            scene_rows.append({"scene_id": scene.scene_id, "scan_id": patient,
                               "scene_family": scene.scene_family,
                               "junction_center_error_mm": center_error, **metrics})
            if len(route["paths_voxel"]) != len(references):
                if route["status"] == "ok":
                    raise ValueError("successful route/reference arm count differs")
                continue
            for arm, (voxel_path, (side, line_id, reference)) in enumerate(zip(route["paths_voxel"], references)):
                predicted = paths.path_stage.voxel_to_lps(np.asarray(voxel_path), affine)
                other, other_id, nonadjacent, nonadjacent_id = anatomy.other_line_distance(
                    patient, side, {int(item[1]) for item in references if item[0] == side}, predicted, cache)
                arm_rows.append({"scene_id": scene.scene_id, "scan_id": patient,
                                 "kind": kind, "arm_index": arm, "true_line_id": line_id,
                                 "path_success": bool(metrics["path_success"]),
                                 **anatomy.distance_metrics(predicted, reference),
                                 "other_line_min_distance_mm": other,
                                 "other_line_id": other_id,
                                 "nonadjacent_line_min_distance_mm": nonadjacent,
                                 "nonadjacent_line_id": nonadjacent_id})
        print(f"train path anatomy: patient {patient} done", flush=True)
        del ct, mask_tree
    scenes_frame = pd.DataFrame(scene_rows)
    arms_frame = pd.DataFrame(arm_rows)
    if len(scenes_frame) != 34 or scenes_frame.scene_id.nunique() != 34:
        raise ValueError("train path scene coverage changed")
    if set(scenes_frame.scan_id) & paths.TEST_IDS:
        raise ValueError("held-out test entered train path audit")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenes_frame.to_csv(args.out_dir / "path_train_scenes.csv", index=False)
    arms_frame.to_csv(args.out_dir / "path_train_arms.csv", index=False)
    summary = {"status": "train_posthoc_anatomy_audit", "held_out_test_accessed": False,
               "selected_config": asdict(config), "slack_fraction": args.slack_fraction,
               "ignore_unavoidable_endpoint_contact": args.ignore_unavoidable_endpoint_contact,
               "avoid_other_endpoints": args.avoid_other_endpoints,
               "scenes": len(scenes_frame),
               "topological_success": int(scenes_frame.path_success.sum()),
               "by_kind": {kind: {
                   "arms": len(group),
                   "mean_route_within_0p75mm": float(group.route_within_0p75mm_fraction.mean()),
                   "mean_true_covered_0p75mm": float(group.true_covered_0p75mm_fraction.mean()),
                   "median_route_over_true_arc_length": float(group.route_over_true_arc_length.median()),
                   "near_nonadjacent_line_0p75mm": int(group.nonadjacent_line_min_distance_mm.le(0.75).sum()),
               } for kind, group in arms_frame.groupby("kind")},
               "junction_center_error_mm_median": float(scenes_frame.junction_center_error_mm.median()),
               "source_sha256": {"scene_plan": graph.sha256(paths.SCENES / "scene_plan_train_val.csv"),
                                 "chosen_config": graph.sha256(CHOSEN)}}
    (args.out_dir / "path_train_anatomy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
