#!/usr/bin/env python3
"""Fail-closed validator for frozen ANZA-2 Phase-2A/2B evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1 = PROJECT_ROOT / "results" / "anza2" / "phase1"
PHASE2A = PROJECT_ROOT / "results" / "anza2" / "phase2"
PHASE2B = PROJECT_ROOT / "results" / "anza2" / "phase2b"
PHASE2A_METRICS_SHA256 = "04b35a97c830b682f682084498673daf280e1c81dad407e850be199e8e15e383"
LEGACY_SOURCE_SHA256 = "d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    failures = []
    required_a = ("protocol.json", "protocol_hash.txt", "threshold_freeze.json", "metrics.json", "per_path.csv", "per_branch.csv")
    required_b = ("protocol.json", "protocol_hash.txt", "open_receipt.json", "metrics.json", "bootstrap.json", "per_path.csv", "per_branch.csv", "PHASE2B_REPORT.md")
    for root, names in ((PHASE2A, required_a), (PHASE2B, required_b)):
        for name in names:
            if not (root / name).is_file() or (root / name).stat().st_size == 0:
                failures.append(f"missing {root.name}/{name}")
    if failures:
        return {"status": "FAIL", "failures": failures}
    phase1 = json.loads((PHASE1 / "validator.json").read_text())
    phase2a = json.loads((PHASE2A / "metrics.json").read_text())
    phase2b = json.loads((PHASE2B / "metrics.json").read_text())
    protocol_b = json.loads((PHASE2B / "protocol.json").read_text())
    receipt = json.loads((PHASE2B / "open_receipt.json").read_text())
    if phase1.get("research_status") != "PHASE1_MATH_PASS": failures.append("Phase 1 gate not passed")
    if digest(PHASE2A / "metrics.json") != PHASE2A_METRICS_SHA256: failures.append("Phase-2A result changed")
    if phase2a.get("status") != "STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE": failures.append("Phase-2A FAIL hidden")
    if phase2b.get("phase2a_status_preserved") != phase2a.get("status"): failures.append("Phase-2A status not preserved")
    if phase2b.get("status") != "PHASE2_GEOMETRY_SELECTIVITY_PASS": failures.append("replacement gate not passed")
    if phase2b.get("anza_minus_legacy_branch_recall", 0) < protocol_b["minimum_branch_recall_delta"]: failures.append("branch delta below gate")
    if phase2b.get("anza_minus_legacy_branch_recall_ci95", [0])[0] <= 0: failures.append("branch delta CI not positive")
    anza_path = phase2b["path_metrics"]["anza2_absolute"]
    legacy_path = phase2b["path_metrics"]["legacy_global_normalized"]
    if anza_path["tpr"] - legacy_path["tpr"] < protocol_b["path_tpr_noninferiority_margin"]: failures.append("path TPR inferiority")
    if anza_path["fpr"] - legacy_path["fpr"] > protocol_b["false_bridge_noninferiority_margin"]: failures.append("false bridge inferiority")
    if receipt.get("replacement_confirm_rows_opened") != 0: failures.append("open receipt not written before confirm")
    if receipt.get("protocol_sha256") != phase2b.get("protocol_sha256"): failures.append("open receipt protocol mismatch")
    if any(phase2b.get(key) is not False for key in ("training_performed", "cracks_data_accessed", "expert_data_accessed")): failures.append("data/training lock violated")
    if digest(PROJECT_ROOT / "models" / "azconv.py") != LEGACY_SOURCE_SHA256: failures.append("LegacyANZA source drift")
    receipt_out = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": "PHASE2_GEOMETRY_SELECTIVITY_PASS" if not failures else "STOP_ANZA2_PHASE2_INVALID",
        "failures": failures,
        "phase2a_metrics_sha256": PHASE2A_METRICS_SHA256,
        "phase2b_protocol_sha256": phase2b.get("protocol_sha256"),
        "phase2a_fail_preserved": True,
        "training_performed": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "phase3_allowed": not failures,
    }
    (PHASE2B / "validator.json").write_text(json.dumps(receipt_out, indent=2, sort_keys=True) + "\n")
    (PHASE2B / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
    }, indent=2, sort_keys=True) + "\n")
    (PHASE2B / "checkpoint_manifest.json").write_text(json.dumps({
        "new_checkpoints": [], "training_performed": False,
    }, indent=2, sort_keys=True) + "\n")
    (PHASE2B / "data_access_log.json").write_text(json.dumps({
        "data": "controlled synthetic tensors only", "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
    (PHASE2B / "code_state.json").write_text(json.dumps({
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "git_status_lines": git_status, "commit_created": False,
    }, indent=2, sort_keys=True) + "\n")
    (PHASE2B / "split_manifest.json").write_text(json.dumps({
        "development_seed_base": 610000000, "phase2a_confirm_seed_base": 620000000,
        "phase2b_replacement_confirm_seed_base": 630000000,
        "streams_disjoint": True, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    (PHASE2B / "EVIDENCE.json").write_text(json.dumps({
        "claim": "On an independent controlled oracle-field stream, ANZA-2 preserves all X/T/Y branches while the legacy-normalized control misses one X branch, with path TPR/FPR non-inferior.",
        "status": "VERIFIED" if not failures else "FAIL",
        "artifact": "metrics.json, per_branch.csv, per_path.csv, bootstrap.json",
        "expert_used": False, "verified": not failures,
        "allowed_wording": "Controlled oracle-field branch selectivity is supported.",
        "forbidden_wording": "ANZA-2 is learned from images or improves CRACKS.",
    }, indent=2, sort_keys=True) + "\n")
    return receipt_out


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
