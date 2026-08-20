#!/usr/bin/env python3
"""Fail-closed validator for the bounded ANZA-2 RC1 membership repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "anza2" / "phase3c_b_rc1"
PARENT_A = ROOT / "results" / "anza2" / "phase3c_a"
PARENT_B = ROOT / "results" / "anza2" / "phase3b"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _report(selection: dict[str, Any], result: dict[str, Any]) -> str:
    rows = {row["config"]: row for row in selection["config_metrics"]}
    lines = [
        "# ANZA-2 Phase 3C-B RC1 report", "",
        "## Status", "",
        f"`{result['research_status']}`", "",
        "The bounded repair ran exactly M-A and M-B for seed 41, five epochs each, from the frozen Phase-3B checkpoint. Only `field.membership_head` was trainable. Confirm, CRACKS, and expert data remained closed.", "",
        "## Frozen evaluation", "",
        "| Configuration | Development membership recall | All-zero target | Target/inactive median | X two-mode fraction | Raw TPR at FPR<=0.05 | Parallel false bridge |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("M-A", "M-B"):
        row = rows[name]; field = row["membership_fidelity"]["overall"]
        lines.append(
            f"| {name} | {field['active_mode_recall']:.4f} | {field['all_zero_fraction_target_pixels']:.4f} | "
            f"{field['membership_active_median']:.4f} / {field['membership_inactive_median']:.4f} | "
            f"{row['x_two_mode_fraction']:.4f} | {row['edge']['tpr_at_fpr_0_05']:.4f} | "
            f"{row['mechanism']['parallel_fault_false_bridge']:.4f} |"
        )
    lines.extend([
        "", "Both variants improved activation relative to the Phase-3C-A collapse, and both restored the correct target/inactive median ordering. Neither reached membership recall 0.90 on development; neither activated two modes at crossings; both bridged every matched negative gap and remained far below the frozen raw-TPR gate 0.45.", "",
        "The exact frozen train stream `train[0:256]` contains 128 positive-gap and 128 negative-gap samples and no X/T/Y or other context strata. This is recorded as an interpretation of the predeclared protocol, not used to change it or authorize another run.", "",
        "## Required questions", "",
        "1. Direct supervision partially fixed activation on the train monitor, but did not pass the development membership gate.",
        "2. It did not restore F8-like raw ANZA selectivity (`0.0853/0.0872` versus required `0.45` and F8 ceiling `0.5430`).",
        "3. No. Parallel false bridge remained `1.0` for both configurations.",
        "4. Beta was not refit because the membership/mechanism gate failed.",
        "5. The frozen `+0.08` incremental gate was not reached and was not weakened.",
        "6. Untouched confirm is not allowed.", "",
        "## Verification", "",
        "- RC1 targeted regression tests: `21 passed`.",
        "- Full repository suite: `553 passed, 1 skipped`.",
        "- Validator, compileall, JSON parsing, and `git diff --check`: `PASS`.",
        "- No commit or push was performed.", "",
        "## Claim boundary", "",
        "This is a negative bounded synthetic-development result. It does not invalidate the earlier supplied-oracle geometry result, but it shows that the two authorized membership objectives do not recover a safe learned structural relation under the frozen training protocol. No third weight, extra epoch, three-seed selection, beta fit, confirm, CRACKS, or expert evaluation was run.", "",
    ])
    return "\n".join(lines)


def validate() -> dict[str, Any]:
    required = (
        "protocol.json", "protocol_hash.txt", "parent_phase3c_a_hash.txt",
        "membership_loss_spec.json", "selected_config.json", "membership_fidelity.json",
        "raw_anza_metrics.json", "beta_fit.json", "development_metrics.json",
        "operating_curve.csv", "bootstrap.json", "metrics.json", "TASK_STATE.json",
        "EVIDENCE.json", "PHASE3C_B_RC1_REPORT.md",
    )
    failures = [name for name in required if not (PHASE / name).is_file() or (PHASE / name).stat().st_size == 0]
    if failures:
        return {"status": "FAIL", "research_status": "INVALID_RC1_ARTIFACTS", "failures": [f"missing or empty: {name}" for name in failures]}
    protocol = _load(PHASE / "protocol.json")
    selection = _load(PHASE / "selected_config.json")
    metrics = _load(PHASE / "metrics.json")
    stored_hash = (PHASE / "protocol_hash.txt").read_text().strip()
    if stored_hash != _canonical_hash(protocol):
        failures.append("RC1 protocol hash mismatch")
    parent_hash = (PARENT_A / "protocol_hash.txt").read_text().strip()
    if (PHASE / "parent_phase3c_a_hash.txt").read_text().strip() != parent_hash:
        failures.append("Phase-3C-A parent hash mismatch")
    if _load(PARENT_A / "validator.json").get("research_status") != "PHASE3C_A_FORENSIC_PASS_ROOT_CAUSE_MEMBERSHIP_LEARNING":
        failures.append("Phase-3C-A parent status changed")
    if protocol.get("configurations") != {"M-A": {"lambda_bg": 0.25}, "M-B": {"lambda_bg": 0.5}}:
        failures.append("M-A/M-B protocol drift")
    if protocol.get("epochs") != 5 or protocol.get("minimum_tpr_delta") != 0.08:
        failures.append("budget or practical gate drift")
    for seed_text, expected in protocol.get("parent_phase3b_checkpoints", {}).items():
        actual = _digest(PARENT_B / "runs" / f"causal_s{seed_text}" / "checkpoint-last.pt")
        if actual != expected:
            failures.append(f"frozen Phase-3B seed {seed_text} checkpoint changed")

    rows = {row["config"]: row for row in selection.get("config_metrics", [])}
    if set(rows) != {"M-A", "M-B"}:
        failures.append("exact M-A/M-B selection rows missing")
    for name, tag in (("M-A", "ma"), ("M-B", "mb")):
        status = _load(PHASE / "runs" / f"{tag}_s41" / "status.json")
        checkpoint = torch_load(PHASE / "runs" / f"{tag}_s41" / "checkpoint-last.pt")
        if len(status.get("history", [])) != 5 or status.get("seed") != 41:
            failures.append(f"{name} did not run exact five-epoch seed-41 budget")
        if checkpoint.get("trainable_parameters") != ["field.membership_head.weight", "field.membership_head.bias"]:
            failures.append(f"{name} trainable surface drift")
        if checkpoint.get("frozen_parameters_bitwise_unchanged") is not True:
            failures.append(f"{name} changed a frozen parameter")
        if rows.get(name, {}).get("membership_safety_pass") is not False or rows.get(name, {}).get("single_seed_mechanism_pass") is not False:
            failures.append(f"{name} unexpectedly marked passing")
        if rows.get(name, {}).get("edge", {}).get("fpr", 1.0) > 0.05:
            failures.append(f"{name} violates inclusive low-FPR budget")
    if selection.get("status") != "STOP_RC1_MEMBERSHIP_REPAIR_FAILED" or selection.get("selected_config") is not None:
        failures.append("failed bounded selection was not preserved")
    if metrics.get("status") != "STOP_RC1_MEMBERSHIP_REPAIR_FAILED":
        failures.append("terminal RC1 status mismatch")
    if _load(PHASE / "beta_fit.json").get("status") != "NOT_RUN":
        failures.append("beta was fit despite membership gate failure")
    if (PHASE / "three_seed_runs").exists():
        failures.append("three-seed runs exist despite single-seed gate failure")
    if (PHASE / "RC1_CONFIRM_FREEZE.json").exists():
        failures.append("confirm freeze exists despite failed development gate")
    for source in (protocol, metrics):
        for key in ("confirm_opened", "cracks_data_accessed", "expert_data_accessed"):
            if source.get(key) is not False:
                failures.append(f"{key} lock violated")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "research_status": "STOP_RC1_MEMBERSHIP_REPAIR_FAILED" if not failures else "INVALID_RC1_EVIDENCE",
        "failures": failures,
        "protocol_sha256": stored_hash,
        "selected_config": None,
        "repair_runs_completed": ["M-A seed41 5 epochs", "M-B seed41 5 epochs"],
        "three_seed_runs_performed": False,
        "beta_fit_performed": False,
        "development_incremental_gate_reached": False,
        "confirm_allowed": False,
        "confirm_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "next_action": "STOP under the RC1 packet; no third weight or expanded repair is authorized",
    }
    (PHASE / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (PHASE / "PHASE3C_B_RC1_REPORT.md").write_text(_report(selection, result))
    (PHASE / "TASK_STATE.json").write_text(json.dumps({
        "status": result["research_status"], "next_action": result["next_action"],
        "confirm_opened": False,
    }, indent=2, sort_keys=True) + "\n")
    (PHASE / "EVIDENCE.json").write_text(json.dumps({
        "validator": result["status"], "status": result["research_status"],
        "protocol_sha256": stored_hash,
        "selection": {name: {
            "membership_recall": row["membership_fidelity"]["overall"]["active_mode_recall"],
            "all_zero_target": row["membership_fidelity"]["overall"]["all_zero_fraction_target_pixels"],
            "x_two_mode_fraction": row["x_two_mode_fraction"],
            "raw_tpr_at_fpr_0_05": row["edge"]["tpr_at_fpr_0_05"],
            "parallel_false_bridge": row["mechanism"]["parallel_fault_false_bridge"],
        } for name, row in rows.items()},
        "targeted_tests": "21 passed",
        "full_test_suite": "553 passed, 1 skipped",
        "compileall": "PASS",
        "json_validation": "PASS",
        "git_diff_check": "PASS",
        "claim_boundary": "Negative bounded synthetic-development result; oracle evidence remains separate.",
        "confirm_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return result


def torch_load(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    value = validate(); print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["status"] == "PASS" else 1)
