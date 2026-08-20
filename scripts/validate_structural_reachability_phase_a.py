#!/usr/bin/env python3
"""Fail-closed validator for the frozen Structural Reachability Phase A."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural_reachability.phase_a import (
    BOOTSTRAP_RESAMPLES,
    FPR_MAX,
    OUTPUT_ROOT,
    RELATIONS,
    SEEDS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_phase_a(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    required = (
        "protocol.json", "protocol_hash.txt", "split_manifest.json", "data_access_log.json",
        "per_candidate.csv", "edge_features.csv", "operating_curve.csv", "bootstrap_summary.json",
        "metrics.json", "calibration.json", "checkpoint_manifest.json", "code_state.json",
        "environment.json", "EVIDENCE.json", "PHASE_A_GEOMETRY_PROBE_REPORT.md",
        "confuser_metrics.json", "SECONDARY_CONFUSER_AUDIT.md",
        "fig_low_fpr_geometry.png", "fig_low_fpr_geometry.svg",
    )
    for name in required:
        path = root / name
        check(path.is_file() and path.stat().st_size > 0, f"missing or empty {name}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    protocol = json.loads((root / "protocol.json").read_text())
    protocol_hash = (root / "protocol_hash.txt").read_text().strip()
    metrics = json.loads((root / "metrics.json").read_text())
    evidence = json.loads((root / "EVIDENCE.json").read_text())
    split = json.loads((root / "split_manifest.json").read_text())
    access = json.loads((root / "data_access_log.json").read_text())
    bootstrap = json.loads((root / "bootstrap_summary.json").read_text())
    confusers = json.loads((root / "confuser_metrics.json").read_text())
    checkpoints = json.loads((root / "checkpoint_manifest.json").read_text())

    check(_canonical_hash(protocol) == protocol_hash, "protocol hash mismatch")
    check(metrics.get("protocol_sha256") == protocol_hash, "metrics protocol hash mismatch")
    check(evidence.get("protocol_sha256") == protocol_hash, "evidence protocol hash mismatch")
    check(evidence.get("metrics_sha256") == _sha256(root / "metrics.json"), "metrics evidence hash mismatch")
    check(evidence.get("per_candidate_sha256") == _sha256(root / "per_candidate.csv"), "candidate evidence hash mismatch")
    check(protocol.get("training_performed") is False, "protocol permits training")
    check(metrics.get("training_performed") is False, "metrics claim training")
    check(evidence.get("training_performed") is False, "evidence claims training")
    check(protocol.get("expert_data_accessed") is False, "protocol accessed expert")
    check(metrics.get("expert_data_accessed") is False, "metrics accessed expert")
    check(metrics.get("expert_scores_used") is False, "metrics used expert scores")
    check(evidence.get("expert_data_accessed") is False, "evidence accessed expert")
    check(access.get("expert_data_accessed") is False and access.get("expert_paths") == [], "access log is not expert-locked")
    check(access.get("crowd_annotations_opened_during_phase_a") == [], "Phase A unexpectedly reopened annotations")
    check(confusers.get("primary_gate_unchanged") is True, "confuser audit changed the primary gate")
    check(confusers.get("primary_status") == metrics.get("status"), "confuser audit primary status drift")
    check(confusers.get("pair_count") == 20, "model-generated confuser count drift")
    check(confusers.get("expert_data_accessed") is False, "confuser audit accessed expert")
    check(confusers.get("training_performed") is False, "confuser audit performed training")
    check(protocol.get("seeds") == list(SEEDS), "seed set drift")
    check(protocol.get("relations") == list(RELATIONS), "relation matrix drift")
    check(float(protocol["bootstrap"]["resamples"]) == BOOTSTRAP_RESAMPLES, "bootstrap budget drift")
    check(split.get("section_disjoint") is True, "source split is not section-disjoint")
    check(split.get("validation_pair_count") == 120, "validation pair count drift")
    check(split.get("validation_section_count") == 73, "validation section count drift")
    check(split.get("expert_section_ids") == [], "expert section IDs entered Phase A")
    check(set(checkpoints) == {str(seed) for seed in SEEDS}, "checkpoint seed manifest drift")
    for row in checkpoints.values():
        path = Path(row["path"])
        check(path.is_file() and _sha256(path) == row["sha256"], f"checkpoint hash drift: {path}")

    rows = list(csv.DictReader((root / "per_candidate.csv").open()))
    check(len(rows) == 120 * 2 * len(SEEDS) * len(RELATIONS), "candidate row count drift")
    check({int(row["section_id"]) for row in rows} == set(split["validation_section_ids"]), "candidate sections drift")
    check({int(row["seed"]) for row in rows} == set(SEEDS), "candidate seeds drift")
    check({row["relation"] for row in rows} == set(RELATIONS), "candidate relations drift")
    check({int(row["label"]) for row in rows} == {0, 1}, "candidate labels are not balanced binary")
    check(all(math.isfinite(float(row["score"])) and 0.0 <= float(row["score"]) <= 1.0 for row in rows), "invalid candidate score")
    for seed in SEEDS:
        for relation in RELATIONS:
            selected = [row for row in rows if int(row["seed"]) == seed and row["relation"] == relation]
            check(sum(int(row["label"]) for row in selected) == 120 and len(selected) == 240, f"unbalanced {relation} seed {seed}")

    baseline = [
        float(metrics["relations"][RELATIONS[0]]["per_seed"][str(seed)]["tpr_at_fpr_max"])
        for seed in SEEDS
    ]
    mean = sum(baseline) / len(baseline)
    expected_sd = math.sqrt(sum((value - mean) ** 2 for value in baseline) / (len(baseline) - 1))
    expected_delta = max(0.05, 2.0 * expected_sd)
    check(math.isclose(float(metrics["baseline_seed_sd"]), expected_sd, abs_tol=1e-12), "baseline seed SD drift")
    check(math.isclose(float(metrics["delta_A"]), expected_delta, abs_tol=1e-12), "delta_A was not frozen from baseline variability")
    primary = metrics["primary_comparison"]
    partial = metrics["partial_auc_comparison"]
    for result in (primary, partial, bootstrap["tpr"], bootstrap["low_fpr_partial_auc"]):
        check(result.get("resamples") == BOOTSTRAP_RESAMPLES, "bootstrap resample count drift")
        check(result.get("resampling_unit") == "section_id", "bootstrap did not use sections")
        check(result.get("section_count") == 73, "bootstrap section count drift")
    check(primary.get("metric") == "tpr_at_fpr_max", "primary metric changed")
    check(partial.get("metric") == "low_fpr_partial_auc_normalized", "partial-AUC metric changed")
    check(float(metrics["relations"][RELATIONS[0]]["seed_mean"]["achieved_fpr"]) <= FPR_MAX + 1e-12, "baseline exceeds FPR budget")
    check(float(metrics["relations"][RELATIONS[-1]]["seed_mean"]["achieved_fpr"]) <= FPR_MAX + 1e-12, "ANZA exceeds FPR budget")

    expected_checks = {
        "tpr_delta_meaningful": float(primary["point_delta"]) >= float(metrics["delta_A"]),
        "tpr_delta_ci_low_positive": float(primary["ci95"][0]) > 0.0,
        "low_fpr_partial_auc_improved": float(partial["point_delta"]) > 0.0,
    }
    check(metrics.get("gate_checks") == expected_checks, "gate checks do not match frozen rule")
    passed = all(expected_checks.values())
    expected_status = "PHASE_A_PASS" if passed else "STOP_ARCHITECTURAL_ANZA_NO_CAUSAL_GEOMETRY_GAIN"
    check(metrics.get("status") == expected_status, "Phase-A status does not match gate")
    check(metrics.get("phase_b_authorized") is passed, "Phase-B authorization does not match gate")
    check((root / "FAILURE_ANALYSIS.md").is_file() if not passed else True, "negative gate lacks failure analysis")
    report = (root / "PHASE_A_GEOMETRY_PROBE_REPORT.md").read_text()
    check(expected_status in report, "report status drift")
    check("EXPERT ACCESSED: no" in report and "TRAINING PERFORMED: no" in report, "report lock disclosure missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "research_status": expected_status,
        "protocol_sha256": protocol_hash,
        "phase_b_authorized": passed,
        "expert_data_accessed": False,
        "training_performed": False,
        "validated_candidate_rows": len(rows),
        "validated_sections": 73,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def main() -> int:
    receipt = validate_phase_a()
    if receipt["status"] == "PASS":
        (OUTPUT_ROOT / "VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
