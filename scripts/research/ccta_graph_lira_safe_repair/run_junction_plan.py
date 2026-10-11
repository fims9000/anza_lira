#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_junction_plan import generate_candidate_plan


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/research/ccta_graph_lira_safe_repair/junction_plan.json"


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["generator_status"] != "frozen_train_val_v1":
        raise ValueError("JUNCTION generator config is not frozen")
    if set(config["splits"]) != {"train", "val"}:
        raise ValueError("JUNCTION plan is restricted to train/val")

    perturbations = config["perturbations"]
    if perturbations != {
        "enabled": False,
        "position_jitter_mm": 0.0,
        "tangent_noise_deg": 0.0,
    }:
        raise ValueError("Perturbations are not implemented in this stage")

    cut = config["cut"]
    if cut["mode"] != "adaptive":
        raise ValueError("This protocol expects adaptive cut mode")

    candidate_filter = config["candidate_filter"]
    if candidate_filter["supported_degrees"] != [3, 4]:
        raise ValueError("This protocol must support degree-3 and degree-4 junctions")

    negatives = config["negative_sampling"]
    expected_negative_rules = {
        "enabled": True,
        "strategy": "one_and_two_arm_replacement",
        "same_scan_only": True,
        "same_side_only": True,
        "exclude_incident_polylines": True,
        "segment_label_role": "audit_only",
        "tangent_orientations": "both",
    }
    actual_negative_rules = {key: negatives[key] for key in expected_negative_rules}
    if actual_negative_rules != expected_negative_rules:
        raise ValueError("Unsupported negative sampling rules")

    matching = config["geometry_matching"]
    expected_matching_rules = {
        "enabled": True,
        "reference": "source_positive",
        "feature_scaling": "train_positive_standard_deviation",
        "selection": "nearest_per_source_positive",
    }
    if matching != expected_matching_rules:
        raise ValueError("Unsupported geometry matching rules")

    out_dir = resolve_repo_path(config["out_dir"])
    selection = config["primary_selection"]
    if selection["degree"] != 3:
        raise ValueError("Primary protocol is restricted to degree-3")
    if selection["quota_scope"] != "per_positive_cap":
        raise ValueError("Primary selection must use per-positive caps")
    if selection["fill_local_shortage_from_other_junctions"]:
        raise ValueError("Negatives cannot be reassigned across junctions")

    audit_rows, candidate_rows, endpoint_rows, raw_negative_count = (
        generate_candidate_plan(
            source_root=resolve_repo_path(config["source_root"]),
            out_dir=out_dir,
            target_distance_mm=float(cut["target_distance_mm"]),
            max_arm_fraction=float(cut["max_arm_fraction"]),
            max_span_mm=float(candidate_filter["max_endpoint_span_mm"]),
            decoy_step_mm=float(negatives["decoy_sample_step_mm"]),
            decoy_endpoint_margin_mm=float(
                negatives["decoy_endpoint_margin_mm"]
            ),
            one_arm_per_positive=int(
                selection["one_arm_negatives_per_positive"]
            ),
            two_arm_per_positive=int(
                selection["two_arm_negatives_per_positive"]
            ),
        )
    )

    positive_count = sum(int(row["y"]) == 1 for row in candidate_rows)
    negative_count = sum(int(row["y"]) == 0 for row in candidate_rows)
    development_positive_recall = sum(
        int(row["build_success"]) and int(row["span_eligible"]) for row in audit_rows
    ) / len(audit_rows)
    positive_fraction = positive_count / len(candidate_rows)
    counts_by_degree_and_label = Counter(
        (int(row["degree"]), int(row["y"])) for row in candidate_rows
    )
    selected_by_junction_and_type = Counter(
        (str(row["source_junction_id"]), int(row["n_replaced_arms"]))
        for row in candidate_rows
        if row["y"] == 0
    )
    positive_junction_ids = {
        str(row["source_junction_id"]) for row in candidate_rows if row["y"] == 1
    }
    selected_junction_ids = {key[0] for key in selected_by_junction_and_type}
    no_negative_junctions = sorted(positive_junction_ids - selected_junction_ids)
    source_groups = {
        str(row["source_junction_id"]): str(row["benchmark_group"])
        for row in candidate_rows
        if row["y"] == 1
    }
    group_counts = Counter(source_groups.values())
    if set(group_counts) - {"competitive", "isolated"}:
        raise RuntimeError("Unknown benchmark group")
    if any(
        row["benchmark_group"] != source_groups[str(row["source_junction_id"])]
        for row in candidate_rows
    ):
        raise RuntimeError("Candidate/source benchmark group mismatch")
    if any(
        source in selected_junction_ids
        for source, group in source_groups.items()
        if group == "isolated"
    ):
        raise RuntimeError("Isolated source has a selected false candidate")
    match_distances = np.asarray(
        [
            float(row["geometry_match_distance"])
            for row in candidate_rows
            if row["y"] == 0
        ],
        dtype=float,
    )
    distance_quantiles = np.quantile(match_distances, [0.0, 0.25, 0.5, 0.75, 1.0])
    if development_positive_recall != 1.0:
        raise RuntimeError(
            "Development positive recall is "
            f"{development_positive_recall:.6f}, expected 1.0"
        )
    if any(row["split"] == "test" for row in candidate_rows):
        raise RuntimeError("Held-out test row entered development candidate plan")
    for split, expected_ids in config["expected_patient_ids"].items():
        actual_ids = {
            int(row["scan_id"]) for row in candidate_rows if row["split"] == split
        }
        if actual_ids != set(map(int, expected_ids)):
            raise RuntimeError(f"Unexpected frozen patient IDs for {split}")
    if any(
        float(row["span_mm"]) > candidate_filter["max_endpoint_span_mm"]
        for row in candidate_rows
    ):
        raise RuntimeError("Candidate above maximum endpoint span")
    if any(row["degree"] != 3 for row in candidate_rows):
        raise RuntimeError("Non-degree-3 row entered primary plan")
    caps = {
        1: int(selection["one_arm_negatives_per_positive"]),
        2: int(selection["two_arm_negatives_per_positive"]),
    }
    if any(
        count > caps[n_replaced]
        for (_, n_replaced), count in selected_by_junction_and_type.items()
    ):
        raise RuntimeError("Per-positive negative cap exceeded")

    print(f"junctions processed: {len(audit_rows)}")
    print(f"junctions built: {sum(int(row['build_success']) for row in audit_rows)}")
    print(f"junctions span-eligible: {sum(int(row['span_eligible']) for row in audit_rows)}")
    print(f"primary positive candidates: {positive_count}")
    print(f"primary negative candidates: {negative_count}")
    print(f"raw negative candidates: {raw_negative_count}")
    print(f"degree-4 development junctions: {sum(int(row['degree']) == 4 for row in audit_rows)}")
    print(f"development positive recall: {development_positive_recall:.4f}")
    print(f"positive fraction: {positive_fraction:.4f}")
    print(f"candidate endpoints: {len(endpoint_rows)}")
    print(f"positive sources without negatives: {len(no_negative_junctions)}")
    print(f"benchmark groups: {dict(sorted(group_counts.items()))}")
    print(
        "geometry-match distance q0/q25/q50/q75/q100: "
        + "/".join(f"{value:.4f}" for value in distance_quantiles)
    )
    print(f"degree/label counts: {dict(sorted(counts_by_degree_and_label.items()))}")
    print("warning: degree-4 is not estimable on train/val under this candidate rule")
    print(f"output: {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
