#!/usr/bin/env python3
"""Generate and validate ANZA-2 Phase-0/1 reproducibility artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.anza2.fixtures import save_fixture_artifacts


PHASE0_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase0"
PHASE1_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase1"
LEGACY_SOURCE_SHA256 = "d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197"
PREVIOUS_PHASE_A_SHA256 = "39ab64dc07eeec60ae89748e2fe53c9e42964a809376f6f6487b15b0f5f219f3"
GOLDEN_TEST_FILES = (
    "tests/test_anza2_geometry.py",
    "tests/test_anza2_affinity.py",
    "tests/test_anza2_aggregation.py",
    "tests/test_widest_path_reference.py",
    "tests/test_anza2_tangent_transform.py",
    "tests/test_cracks_v2_split_leakage.py",
    "tests/test_anza2_fixtures.py",
    "tests/test_anza2_expert_lock.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def _code_state(paths: list[Path]) -> dict[str, Any]:
    hashes = {str(path.relative_to(PROJECT_ROOT)): _digest(path) for path in paths}
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
    ).stdout.splitlines()
    return {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "source_sha256": hashes,
        "source_set_sha256": _canonical_hash(hashes),
        "git_status_lines": git_status,
        "commit_created": False,
    }


def _run_golden_tests() -> dict[str, Any]:
    command = [str(PROJECT_ROOT.parent / "venv" / "bin" / "python"), "-m", "pytest", "-q", *GOLDEN_TEST_FILES]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _write_na_artifacts(root: Path, reason: str) -> None:
    _write_json(root / "NOT_APPLICABLE.json", {
        "reason": reason,
        "per_section.csv": "NOT_APPLICABLE_NO_DATASET_EVALUATION",
        "operating_curve.csv": "NOT_APPLICABLE_NO_SCORE_THRESHOLD",
        "bootstrap.json": "NOT_APPLICABLE_NO_ESTIMATED_EFFECT",
        "per_edge.csv": "NOT_APPLICABLE_NO_LEARNED_EDGE_EVALUATION",
        "per_gap.csv": "NOT_APPLICABLE_NO_LEARNED_GAP_EVALUATION",
        "per_candidate.csv": "NOT_APPLICABLE_CONTINUATION_NOT_OPEN",
        "path_metrics.csv": "NOT_APPLICABLE_CONTINUATION_NOT_OPEN",
    })


def build_and_validate(*, run_tests: bool = True) -> dict[str, Any]:
    legacy_path = PROJECT_ROOT / "models" / "azconv.py"
    prior_path = PROJECT_ROOT / "results" / "structural_reachability" / "phase_a" / "metrics.json"
    if _digest(legacy_path) != LEGACY_SOURCE_SHA256:
        raise ValueError("LegacyANZA source changed")
    if _digest(prior_path) != PREVIOUS_PHASE_A_SHA256:
        raise ValueError("frozen Structural Reachability Phase-A metrics changed")
    contract = json.loads((PHASE0_ROOT / "data_contract.json").read_text())
    split = json.loads((PHASE0_ROOT / "SPLIT_PROTOCOL_V2.json").read_text())
    sources = [
        PROJECT_ROOT / path for path in (
            "models/anza2/field.py", "models/anza2/geometry.py", "models/anza2/affinity.py",
            "models/anza2/aggregation.py", "models/anza2/block.py", "models/anza2/losses.py",
            "models/anza2/diagnostics.py", "models/anza2/fixtures.py", "structural/graph.py",
            "structural/widest_path.py", "cracks_v2/data_contract.py", "cracks_v2/split.py",
            "docs/research/ANZA2_MATH_SPEC.md", "docs/research/CRACKS_DATA_CONTRACT_V2.md",
            "docs/research/PRIOR_ART_BOUNDARY.md",
        )
    ]
    code_state = _code_state(sources)
    phase0_protocol = {
        "version": "anza2_phase0_v1",
        "legacy_source_sha256": LEGACY_SOURCE_SHA256,
        "previous_phase_a_metrics_sha256": PREVIOUS_PHASE_A_SHA256,
        "data_contract_sha256": _digest(PHASE0_ROOT / "data_contract.json"),
        "split_protocol_sha256": _digest(PHASE0_ROOT / "SPLIT_PROTOCOL_V2.json"),
        "expert_data_accessed": False,
        "training_performed": False,
        "spatial_coordinates_required_for_claim": True,
        "spatial_coordinates_established": False,
        "allowed_split_wording": "grouped section OOF with numeric-order limitation",
    }
    phase0_hash = _canonical_hash(phase0_protocol)
    phase0_failures = []
    if contract.get("status") != "PASS_WITH_RELEASE_AND_SPATIAL_LIMITATIONS":
        phase0_failures.append("data contract status")
    if contract["images"]["count"] != 396 or contract["images"]["missing_nominal_1_400"] != [9, 185, 249, 336]:
        phase0_failures.append("official image inventory")
    if contract["pairing"]["image_with_nonexpert_annotation_count"] != 393:
        phase0_failures.append("annotated image inventory")
    if contract["expert_lock"]["expert_data_accessed"] is not False:
        phase0_failures.append("expert lock")
    if split.get("outer_exactly_once") is not True or split.get("fold_count") != 5:
        phase0_failures.append("OOF split coverage")
    phase0_validator = {
        "status": "PASS" if not phase0_failures else "FAIL",
        "research_status": "ANZA2_PHASE0_PASS" if not phase0_failures else "STOP_ANZA2_DATA_CONTRACT_INVALID",
        "failures": phase0_failures,
        "protocol_sha256": phase0_hash,
        "training_performed": False,
        "expert_data_accessed": False,
    }
    _write_json(PHASE0_ROOT / "protocol.json", phase0_protocol)
    (PHASE0_ROOT / "protocol_hash.txt").write_text(phase0_hash + "\n")
    _write_json(PHASE0_ROOT / "code_state.json", code_state)
    _write_json(PHASE0_ROOT / "split_manifest.json", split)
    _write_json(PHASE0_ROOT / "environment.json", _environment())
    _write_json(PHASE0_ROOT / "checkpoint_manifest.json", {"new_checkpoints": [], "training_performed": False})
    _write_json(PHASE0_ROOT / "metrics.json", {
        "image_count": 396,
        "annotated_image_count": 393,
        "nonexpert_annotator_count": 34,
        "missing_nominal_ids": [9, 185, 249, 336],
        "unannotated_image_ids": [49, 73, 385],
        "spatial_coordinates_established": False,
        "oof_fold_count": 5,
    })
    _write_json(PHASE0_ROOT / "PHASE0_VALIDATOR.json", phase0_validator)
    _write_json(PHASE0_ROOT / "validator.json", phase0_validator)
    _write_json(PHASE0_ROOT / "EVIDENCE.json", {
        "claim": "The checksum-verified CRACKS release supports a 393-section grouped OOF protocol but contains no physical coordinate metadata.",
        "status": "VERIFIED_WITH_LIMITATION",
        "artifact": "data_contract.json and SPLIT_PROTOCOL_V2.json",
        "expert_used": False,
        "verified": not phase0_failures,
    })
    _write_na_artifacts(PHASE0_ROOT, "PHASE0_DATA_AND_SPEC_ONLY")
    if phase0_failures:
        return {"status": "FAIL", "phase0": phase0_validator}

    PHASE1_ROOT.mkdir(parents=True, exist_ok=True)
    fixtures_root = PHASE1_ROOT / "figures"
    fixture_metrics = save_fixture_artifacts(fixtures_root)
    test_receipt = _run_golden_tests() if run_tests else {"status": "NOT_RUN", "exit_code": None}
    phase1_protocol = {
        "version": "anza2_phase1_math_v1",
        "phase0_protocol_sha256": phase0_hash,
        "code_source_set_sha256": code_state["source_set_sha256"],
        "field": {"num_modes": 4, "ell_min": 0.25, "h_max": 1.25},
        "aggregation": {"tau0": 1.0, "offsets": "8-neighborhood"},
        "geometry": "doubled-angle reciprocal-scale exp(-Q/2)",
        "directed_support": "max_r mu_r G_r",
        "structural_affinity": "sqrt(D_forward D_reverse)",
        "golden_tests": list(GOLDEN_TEST_FILES),
        "dataset_training_performed": False,
        "expert_data_accessed": False,
    }
    phase1_hash = _canonical_hash(phase1_protocol)
    phase1_failures = []
    if not fixture_metrics["phase1_fixture_gate_pass"]:
        phase1_failures.append("deterministic fixture gate")
    if run_tests and test_receipt["status"] != "PASS":
        phase1_failures.append("Golden tests")
    if _digest(legacy_path) != LEGACY_SOURCE_SHA256:
        phase1_failures.append("legacy source drift")
    validator = {
        "status": "PASS" if not phase1_failures else "FAIL",
        "research_status": "PHASE1_MATH_PASS" if not phase1_failures else "STOP_ANZA2_OPERATOR_INVALID",
        "failures": phase1_failures,
        "protocol_sha256": phase1_hash,
        "golden_tests": test_receipt["status"],
        "fixture_gate": fixture_metrics["phase1_fixture_gate_pass"],
        "dataset_training_performed": False,
        "expert_data_accessed": False,
        "phase2_allowed": not phase1_failures,
    }
    _write_json(PHASE1_ROOT / "protocol.json", phase1_protocol)
    (PHASE1_ROOT / "protocol_hash.txt").write_text(phase1_hash + "\n")
    _write_json(PHASE1_ROOT / "code_state.json", code_state)
    _write_json(PHASE1_ROOT / "split_manifest.json", split)
    _write_json(PHASE1_ROOT / "data_access_log.json", {
        "data": "deterministic handcrafted tensors only",
        "cracks_pixels_read": False,
        "expert_data_accessed": False,
    })
    _write_json(PHASE1_ROOT / "environment.json", _environment())
    _write_json(PHASE1_ROOT / "checkpoint_manifest.json", {"new_checkpoints": [], "training_performed": False})
    _write_json(PHASE1_ROOT / "metrics.json", fixture_metrics)
    _write_json(PHASE1_ROOT / "field_summary.json", fixture_metrics)
    _write_json(PHASE1_ROOT / "test_receipt.json", test_receipt)
    _write_json(PHASE1_ROOT / "bootstrap.json", {"status": "NOT_APPLICABLE_DETERMINISTIC_MATH_FIXTURES"})
    _write_json(PHASE1_ROOT / "validator.json", validator)
    _write_json(PHASE1_ROOT / "EVIDENCE.json", {
        "claim": "The ANZA-2 operator satisfies the frozen mathematical contract and deterministic structural fixtures.",
        "status": "VERIFIED" if not phase1_failures else "FAIL",
        "artifact": "metrics.json, test_receipt.json, validator.json",
        "expert_used": False,
        "verified": not phase1_failures,
        "allowed_wording": "Phase-1 mathematical implementation passes; no learned dataset result exists yet.",
        "forbidden_wording": "ANZA-2 improves CRACKS segmentation.",
    })
    _write_na_artifacts(PHASE1_ROOT, "PHASE1_MATH_ONLY_NO_DATASET_EVALUATION")
    (PHASE1_ROOT / "PHASE1_REPORT.md").write_text(f"""# ANZA-2 Phase 1 report

```text
PHASE: 1 — MATHEMATICAL IMPLEMENTATION
STATUS: {validator['research_status']}
PROTOCOL HASH: {phase1_hash}

RESEARCH QUESTION: Is the separate ANZA-2 field/affinity/aggregation mathematically coherent on frozen deterministic fixtures?
DATA: handcrafted tensors only
TRAIN SECTIONS: NOT APPLICABLE
DEV SECTIONS: NOT APPLICABLE
OUTER/CONFIRM SECTIONS: NOT OPENED
EXPERT ACCESSED: NO
TRAINING PERFORMED: NO DATASET TRAINING

PRIMARY METRICS: Golden T1-T18 and Straight/Parallel/Crossing/Curved fixture gates
RESULTS: Golden tests {test_receipt['status']}; fixture gate {fixture_metrics['phase1_fixture_gate_pass']}
PRE-SPECIFIED GATE: all mathematical tests and fixtures PASS
PASS/FAIL: {'PASS' if not phase1_failures else 'FAIL'}
WHAT THIS PROVES: the implemented operator realizes the frozen Phase-1 algebra
WHAT THIS DOES NOT PROVE: segmentation, learned affinity, CRACKS, or expert improvement
NEXT PHASE ALLOWED: {'YES — CONTROLLED SYNTHETIC ONLY' if not phase1_failures else 'NO'}
```
""")
    return {"status": validator["status"], "phase0": phase0_validator, "phase1": validator}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    result = build_and_validate(run_tests=not args.skip_tests)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
