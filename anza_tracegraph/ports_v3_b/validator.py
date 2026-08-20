"""Fail-closed validator for SBPP V3-B."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from anza_tracegraph.frozen_source import DENSE_CHECKPOINT
from .repair_data import repair_calibration_hash
from .runner import PROTOCOL, RESULT, _sha, _source_manifest, protocol_hash


def validate() -> dict[str, Any]:
    required = ("protocol.json", "protocol_hash.txt", "source_manifest.json", "split_manifest.json", "old_calibration_forensic.json", "repair_calibration_sweep.csv", "repair_calibration_per_case.csv", "repair_calibration_per_stratum.csv", "repair_calibration_taxonomy.csv", "repair_calibration_recall_vs_k.csv", "repair_calibration_candidate_burden.csv", "sbpp_v3_b_freeze.json", "metrics.json", "zero_training_receipt.json", "TRACEGRAPH_SBPP_V3_B_REPORT.md")
    missing = [name for name in required if not (RESULT / name).is_file()]
    if missing: raise ValueError(f"missing V3-B artifacts: {missing}")
    metrics = json.loads((RESULT / "metrics.json").read_text()); split = json.loads((RESULT / "split_manifest.json").read_text()); freeze = json.loads((RESULT / "sbpp_v3_b_freeze.json").read_text()); receipt = json.loads((RESULT / "zero_training_receipt.json").read_text()); source = json.loads((RESULT / "source_manifest.json").read_text()); sweep = list(csv.DictReader((RESULT / "repair_calibration_sweep.csv").open())); per_case = list(csv.DictReader((RESULT / "repair_calibration_per_case.csv").open()))
    calibration_pass = freeze["calibration_pass"]
    development_required = calibration_pass
    development_files = ("development_per_case.csv", "development_per_stratum.csv", "development_taxonomy.csv", "recall_vs_k.csv")
    selected_eligible = [row for row in metrics["repair_calibration"] if row["eligible"]]
    reference = metrics["hard_reference"]
    independently_eligible = []
    for row in metrics["repair_calibration"]:
        wrong_safe = row["endpoint_close_but_wrong_branch_rate"] <= 1.25 * reference["endpoint_close_but_wrong_branch_rate"] if reference["endpoint_close_but_wrong_branch_rate"] > 0 else row["endpoint_close_but_wrong_branch_rate"] == 0
        if row["branch_recall_at_12"] >= 0.970 and row["median_candidate_branches"] <= 4 and row["p95_candidate_branches"] <= 12 and wrong_safe and row["B6"] == 0: independently_eligible.append(row)
    dev = metrics.get("development"); development_rows = list(csv.DictReader((RESULT / "development_per_case.csv").open())) if development_required else []
    main_names = {"x_crossing_correct", "acute_crossing_correct", "t_junction_continue", "y_junction_continue", "long_gap", "close_parallel_continue", "partial_occlusion_continue"}
    development_gate = not development_required or dev is not None and dev["branch_recall_at_12"] >= 0.95 and dev["median_candidate_branches"] <= 8 and dev["p95_candidate_branches"] <= 16 and dev["taxonomy"]["B6"] == 0 and all(row["branch_recall_at_12"] >= 0.90 for row in metrics["development_per_stratum"] if row["stratum"] in main_names)
    checks = {
        "protocol_hash": protocol_hash() == metrics["protocol_sha256"] == (RESULT / "protocol_hash.txt").read_text().strip(),
        "source_hash": source["sha256"] == _source_manifest()["sha256"],
        "repair_hash": split["repair_calibration"]["hash_frozen_before_evaluation"] and split["repair_calibration"]["sha256"] == freeze["repair_calibration_sha256"] == repair_calibration_hash(),
        "fresh_repair_stream": split["repair_calibration"]["seed"] == 5_231_000_000,
        "complete_sweep": {row["variant"] for row in sweep} == {"hard_reference", "soft_0.30", "soft_0.25", "soft_0.20"} and len(per_case) == 3 * 3840,
        "eligibility_recomputed": {row["tau_s"] for row in selected_eligible} == {row["tau_s"] for row in independently_eligible},
        "highest_eligible_selected": (not independently_eligible and freeze["selected_tau_s"] is None) or freeze["selected_tau_s"] == max(row["tau_s"] for row in independently_eligible),
        "development_gate_lock": all((RESULT / name).is_file() for name in development_files) if development_required else all(not (RESULT / name).exists() for name in development_files),
        "development_sample_size": not development_required or len(development_rows) == 3840 and sum(row["positive"] == "1" for row in development_rows) == 2688,
        "development_gate_recomputed": development_gate,
        "confirm_locked": split["confirm"]["hash_only"] and not split["confirm"]["inference_opened"] and not split["confirm"]["metrics_opened"],
        "checkpoint_immutable": receipt["checkpoint_before"] == receipt["checkpoint_after"] == _sha(DENSE_CHECKPOINT),
        "zero_training": not any(receipt[name] for name in ("training_opened", "optimizer_created", "p0_opened", "transformer_opened", "anza_opened", "path_opened")),
        "downstream_locks": not receipt["confirm_evaluated"] and not receipt["cracks_accessed"] and not receipt["expert_accessed"],
        "status": metrics["status"] in {"SBPP_V3_B_BRANCH_COVERAGE_PASS", "STOP_SBPP_V3_B_SOFT_SUPPORT_FAIL", "STOP_SBPP_V3_B_DEVELOPMENT_FAIL"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError(f"SBPP V3-B validation failed: {failed}")
    return {"validator": "PASS", "research_status": metrics["status"], "checks": checks, "training_opened": False, "confirm_evaluated": False, "cracks_accessed": False, "expert_accessed": False}
