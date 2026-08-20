#!/usr/bin/env python3
"""Fail-closed validator for Original ANZA Phase 0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from original_anza_forensics.audit import (
    LEGACY_SOURCE_SHA256,
    OUTPUT_ROOT,
    PREVIOUS_PHASE_A_METRICS_SHA256,
    SEEDS,
    _canonical_hash,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_phase0(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    required = (
        "protocol.json", "protocol_hash.txt", "operator_forensics.json",
        "split_feasibility.json", "split_manifest.json", "ORIGINAL_ANZA_CONFIRM_SPLIT.json",
        "checkpoint_manifest.json", "data_access_log.json", "environment.json",
        "code_state.json", "FAILURE_ANALYSIS.md", "NOT_APPLICABLE.json",
        "ORIGINAL_ANZA_PHASE0_REPORT.md",
    )
    for name in required:
        path = root / name
        check(path.is_file() and path.stat().st_size > 0, f"missing or empty {name}")
    report = PROJECT_ROOT / "docs" / "research" / "ANZA_OPERATOR_FORENSICS.md"
    decisions = PROJECT_ROOT / "docs" / "research" / "DECISIONS.md"
    check(report.is_file() and report.stat().st_size > 0, "missing operator forensics report")
    check(decisions.is_file() and decisions.stat().st_size > 0, "missing decision log")
    if failures:
        return {"status": "FAIL", "failures": failures}

    protocol = json.loads((root / "protocol.json").read_text())
    protocol_hash = (root / "protocol_hash.txt").read_text().strip()
    result = json.loads((root / "operator_forensics.json").read_text())
    split = json.loads((root / "split_feasibility.json").read_text())
    split_manifest = json.loads((root / "split_manifest.json").read_text())
    confirm_split = json.loads((root / "ORIGINAL_ANZA_CONFIRM_SPLIT.json").read_text())
    access = json.loads((root / "data_access_log.json").read_text())
    checkpoints = json.loads((root / "checkpoint_manifest.json").read_text())
    not_applicable = json.loads((root / "NOT_APPLICABLE.json").read_text())

    check(_canonical_hash(protocol) == protocol_hash, "protocol hash mismatch")
    check(result.get("protocol_sha256") == protocol_hash, "result protocol hash mismatch")
    check(_sha256(PROJECT_ROOT / "models" / "azconv.py") == LEGACY_SOURCE_SHA256, "legacy source changed")
    check(
        _sha256(PROJECT_ROOT / "results" / "structural_reachability" / "phase_a" / "metrics.json")
        == PREVIOUS_PHASE_A_METRICS_SHA256,
        "previous Phase-A result changed",
    )
    check(protocol.get("source_sha256") == LEGACY_SOURCE_SHA256, "protocol source hash drift")
    check(result.get("status") == "STOP_OPERATOR_DEFINITION_MISMATCH", "wrong terminal status")
    check(result["findings"]["membership"]["verdict"] == "MATERIAL_MISMATCH", "membership mismatch missing")
    check(result["findings"]["orientation"]["verdict"] == "MATERIAL_PARAMETERIZATION_MISMATCH", "orientation mismatch missing")
    check(result["findings"]["paper_equation_literal_match"] is False, "literal mismatch not recorded")
    check(result["runtime"]["forward_reconstruction_max_abs_error"] < 1e-6, "forward reconstruction mismatch")
    check(result["runtime"]["membership_sum_max_abs_error"] < 1e-6, "softmax runtime check failed")
    check(result["runtime"]["normalization_sum_max_abs_error"] < 1e-6, "normalization runtime check failed")
    check(result["runtime"]["all_finite"] is True, "runtime tensors are non-finite")
    check(result.get("instrumentation_performed") is False, "instrumentation ran after terminal mismatch")
    check(result.get("confirm_performed") is False, "confirm ran after terminal mismatch")
    check(result.get("training_performed") is False, "training ran in Phase 0")
    check(result.get("expert_data_accessed") is False, "expert accessed in Phase 0")
    check(result.get("expert_scores_used") is False, "expert scores used in Phase 0")
    check(result.get("next_phase_allowed") is False, "next phase incorrectly authorized")
    check(split.get("status") == "STOP_NO_INDEPENDENT_CONFIRM_SPLIT", "split stop missing")
    check(split.get("unseen_image_section_ids") == [49, 73, 385], "unseen section inventory drift")
    check(split.get("unseen_annotation_counts") == {"49": 0, "73": 0, "385": 0}, "unseen annotations appeared")
    check(split.get("eligible_independent_nonexpert_confirm_section_ids") == [], "invalid confirm sections selected")
    check(split_manifest.get("status") == "STOP_NO_INDEPENDENT_CONFIRM_SPLIT", "split manifest stop missing")
    check(split_manifest.get("confirm_section_ids") == [], "split manifest contains invalid confirm sections")
    check(split_manifest.get("confirm_authorized") is False, "split manifest was incorrectly authorized")
    check(confirm_split.get("confirm_authorized") is False, "empty confirm split was authorized")
    check(confirm_split.get("confirm_split_frozen") is False, "empty split was mislabeled frozen")
    check(access.get("image_pixels_read") == [], "image pixels were accessed")
    check(access.get("crowd_annotation_pixels_read") == [], "crowd annotation pixels were accessed")
    check(access.get("expert_paths") == [] and access.get("expert_data_accessed") is False, "expert lock violated")
    code_state = json.loads((root / "code_state.json").read_text())
    check(code_state.get("dirty") is True, "dirty worktree was not recorded")
    check(bool(code_state.get("git_status_lines")), "git status file list was not recorded")
    check(set(checkpoints) == {str(seed) for seed in SEEDS}, "checkpoint manifest seed drift")
    for row in checkpoints.values():
        path = Path(row["path"])
        check(path.is_file() and _sha256(path) == row["sha256"], f"checkpoint hash drift: {path}")
    check(not_applicable.get("metrics.json") == "NOT_APPLICABLE_CONFIRM_NOT_RUN", "missing N/A metric disclosure")
    check(not (root / "ORIGINAL_ANZA_CONFIRM_PROTOCOL.json").exists(), "confirm protocol created after stop")
    check(not (root / "ORIGINAL_ANZA_FORENSIC_CONFIRM_REPORT.md").exists(), "confirm report fabricated")
    check("STOP_OPERATOR_DEFINITION_MISMATCH" in report.read_text(), "forensics report status drift")
    check("STOP_NO_INDEPENDENT_CONFIRM_SPLIT" in decisions.read_text(), "decision log split blocker missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "research_status": "STOP_OPERATOR_DEFINITION_MISMATCH",
        "secondary_split_status": "STOP_NO_INDEPENDENT_CONFIRM_SPLIT",
        "protocol_sha256": protocol_hash,
        "instrumentation_performed": False,
        "confirm_performed": False,
        "training_performed": False,
        "expert_data_accessed": False,
        "next_phase_allowed": False,
    }


if __name__ == "__main__":
    receipt = validate_phase0()
    if receipt["status"] == "PASS":
        (OUTPUT_ROOT / "validator.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["status"] == "PASS" else 1)
