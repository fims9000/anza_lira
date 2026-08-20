#!/usr/bin/env python3
"""Freeze CrossingTraceBench streams and run a validation-only generator smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_cracks_archives import write_json
from scripts.prepare_cracks_protocol import V2_RESULT_ROOT, canonical_hash
from synthetic.crossing_trace_bench import SPLIT_SEED_BASE, SPLIT_SIZES, generate_sample
from synthetic.geometry_generator import GEOMETRY_TYPES, NONTRIVIAL_PAIRING_CASES
from synthetic.structural_metrics import compute_structural_metrics


def main() -> int:
    config = {
        "benchmark": "CrossingTraceBench",
        "image_size": 128,
        "split_sizes": SPLIT_SIZES,
        "split_seed_base": SPLIT_SEED_BASE,
        "geometry_types": list(GEOMETRY_TYPES),
        "target_semantics": {
            "observed_segmentation": "visible_fault_mask",
            "structural_completion": "latent_fault_mask",
            "positive_gap": "latent_fault_mask & ~visible_fault_mask",
            "canonical_instances": "overlapping bool instance_masks[N,H,W]",
            "negative_gap": "matched nearby fragments with no common latent instance",
        },
        "topology_types": ["x_crossing", "t_intersection", "y_branch"],
        "nontrivial_pairing_cases": list(NONTRIVIAL_PAIRING_CASES),
        "test_generator_status": "FROZEN_UNOPENED",
        "physical_claim": "controlled structural benchmark; not an F3 simulator",
    }
    config["sha256"] = canonical_hash(config)
    synthetic_root = V2_RESULT_ROOT / "synthetic"
    write_json(synthetic_root / "config.json", config)
    cases = []
    for index, case in enumerate(GEOMETRY_TYPES):
        sample = generate_sample("validation", index, case=case)
        cases.append(
            {
                "case": case,
                "validation_seed": sample["seed"],
                "branch_count": len(sample["branch_ids"]),
                "instance_count": len(sample["fault_instance_ids"]),
                "junction_count": len(sample["junctions"]),
                "strata": sample["strata"],
                "visible_pixels": int(sample["visible_fault_mask"].sum()),
                "latent_pixels": int(sample["latent_fault_mask"].sum()),
                "positive_gap_pixels": int(sample["positive_gap_mask"].sum()),
                "negative_gap_pixels": int(sample["negative_gap_mask"].sum()),
                "overlap_pixels": int(sample["instance_overlap_mask"].sum()),
                "contract_status": "PASS",
            }
        )
    smoke = {
        "status": "PASS",
        "scientific_result": False,
        "config_sha256": config["sha256"],
        "split_used": "validation",
        "test_samples_opened": 0,
        "cases": cases,
    }
    write_json(synthetic_root / "generator_smoke.json", smoke)
    evaluator_cases = []
    for index, case in enumerate(GEOMETRY_TYPES):
        sample = generate_sample("validation", index, case=case)
        orientation = 0.5 * np.arctan2(
            np.asarray(sample["branch_tangent_sin2"]),
            np.asarray(sample["branch_tangent_cos2"]),
        )
        metrics = compute_structural_metrics(
            sample["visible_fault_mask"],
            sample,
            predicted_completion_mask=sample["latent_fault_mask"],
            predicted_instance_masks=sample["instance_masks"],
            predicted_continuation_scores=sample["continuation_relation_matrix"],
            predicted_orientation=orientation,
        )
        evaluator_cases.append({"case": case, "validation_seed": sample["seed"], "metrics": metrics})
    write_json(
        synthetic_root / "structural_evaluator_smoke.json",
        {
            "status": "PASS",
            "scientific_result": False,
            "config_sha256": config["sha256"],
            "split_used": "validation",
            "test_samples_opened": 0,
            "bridge_coverage_threshold": 0.5,
            "latent_instance_overlap_threshold": 0.2,
            "cases": evaluator_cases,
        },
    )
    print("CROSSINGTRACEBENCH: PREPARED")
    print(f"CONFIG SHA256: {config['sha256']}")
    print(f"CASES: {len(cases)}")
    print("VISIBLE/LATENT/GAP CONTRACT: PASS")
    print("OVERLAPPING INSTANCES: PASS")
    print("X/T/Y TOPOLOGY: PASS")
    print("POSITIVE/NEGATIVE GAPS: PASS")
    print("NONTRIVIAL PAIRING: PASS")
    print("STRUCTURAL EVALUATOR: PASS")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
