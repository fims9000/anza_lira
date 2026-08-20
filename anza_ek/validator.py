"""Fail-closed pre-run and final validators for ANZA-EK E0/E1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .e0_audit import run_e0
from .e1_bench import PAIRS_PER_TASK, TASKS, generate_pair
from .kernels import METHODS, generated_kernel_bank
from .protocol import FREEZE_ROOT, RESULT_ROOT, canonical_hash
from .run_e0_e1 import freeze_inputs, source_manifest


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_pre_run() -> dict[str, Any]:
    protocol, benchmark, receipt = freeze_inputs()
    checks: dict[str, bool] = {}
    checks["protocol_hash_matches"] = receipt["protocol_sha256"] == canonical_hash(protocol)
    checks["benchmark_hash_matches"] = receipt["benchmark_sha256"] == benchmark["sha256"]
    checks["source_hash_matches"] = receipt["source_sha256"] == source_manifest()["sha256"]
    checks["six_predeclared_tasks"] = len(TASKS) == 6 and tuple(benchmark["tasks"]) == TASKS
    checks["pair_count_frozen"] = PAIRS_PER_TASK == 256
    checks["four_controls_frozen"] = tuple(protocol["methods"]) == METHODS and len(METHODS) == 4
    checks["no_training_or_classifier"] = protocol["training_performed"] is False and protocol["learned_classifier"] is False
    checks["downstream_locked"] = not any(protocol[key] for key in ("E2_opened", "conjugacy_opened", "confirm_created", "cracks_accessed", "expert_accessed"))
    first = generate_pair(TASKS[0], 0)
    second = generate_pair(TASKS[0], 0)
    checks["deterministic_pair_generation"] = np.array_equal(first["positive"], second["positive"]) and np.array_equal(first["negative"], second["negative"])
    pixel_equal_count = 0
    minimum_l2 = float("inf")
    for task in TASKS:
        for index in range(PAIRS_PER_TASK):
            pair = generate_pair(task, index)
            pixel_equal_count += int(pair["pixel_equal"])
            minimum_l2 = min(minimum_l2, float(pair["l2_difference"]))
    checks["no_geometry_identical_pairs"] = pixel_equal_count == 0 and minimum_l2 > 1e-6
    kernel_checks = []
    for method in METHODS:
        bank = generated_kernel_bank(method, orientation=0.0, size=int(protocol["kernel_size"]), K=int(protocol["K"]), sigma=float(protocol["seed_sigma"]))
        kernel_checks.append(bank.shape == (7, 65, 65) and np.allclose(bank.mean(axis=(1, 2)), 0.0, atol=1e-12) and np.allclose(np.linalg.norm(bank, axis=(1, 2)), 1.0, atol=1e-12))
    checks["kernel_banks_mean_zero_unit_energy"] = all(kernel_checks)
    e0 = run_e0(grid_size=int(protocol["grid_size_e0"]), K=int(protocol["K"]))
    checks["E0_mathematics_pass"] = e0["status"] == "ANZA_EK_E0_PASS"
    passed = all(checks.values())
    result = {
        "validator_status": "PASS" if passed else "FAIL",
        "research_status": "ANZA_EK_E0_E1_PRE_RUN_PASS" if passed else "ANZA_EK_E0_E1_PRE_RUN_FAIL",
        "run_authorized": passed,
        "protocol_sha256": receipt["protocol_sha256"],
        "benchmark_sha256": receipt["benchmark_sha256"],
        "source_sha256": receipt["source_sha256"],
        "checks": checks,
        "identifiability": {"pair_count": len(TASKS) * PAIRS_PER_TASK, "pixel_equal_count": pixel_equal_count, "minimum_l2_difference": minimum_l2},
        "E0_status": e0["status"],
        "training_performed": False,
        "E2_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    _write(FREEZE_ROOT / "pre_run_validator.json", result)
    if not passed:
        raise ValueError("ANZA-EK E0/E1 pre-run validation failed")
    return result


def validate_final() -> dict[str, Any]:
    metrics_path = RESULT_ROOT / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("ANZA-EK E0/E1 metrics are missing")
    metrics = json.loads(metrics_path.read_text())
    protocol = json.loads((FREEZE_ROOT / "protocol.json").read_text())
    receipt = json.loads((FREEZE_ROOT / "freeze_receipt.json").read_text())
    pre_run = json.loads((FREEZE_ROOT / "pre_run_validator.json").read_text())
    raw_path = RESULT_ROOT / "raw_per_pair.csv"
    row_count = 0
    if raw_path.exists():
        with raw_path.open(newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
    recognized = {"ANZA_EK_E1_MECHANISM_PASS", "STOP_ERGODIC_ANOSOV_LOCAL_FEATURE_NO_MECHANISM", "STOP_ANZA_EK_E0_MATHEMATICAL_VALIDATION_FAIL"}
    e1_present = metrics.get("e1") is not None
    checks = {
        "pre_run_pass": pre_run.get("research_status") == "ANZA_EK_E0_E1_PRE_RUN_PASS",
        "protocol_unchanged": receipt["protocol_sha256"] == canonical_hash(protocol) == metrics.get("protocol_sha256"),
        "source_unchanged": receipt["source_sha256"] == source_manifest()["sha256"] == metrics.get("source_sha256"),
        "benchmark_unchanged": receipt["benchmark_sha256"] == metrics.get("benchmark_sha256"),
        "recognized_status": metrics.get("status") in recognized,
        "E0_pass": metrics.get("e0", {}).get("status") == "ANZA_EK_E0_PASS",
        "all_pair_rows_present": (not e1_present) or row_count == len(METHODS) * len(TASKS) * PAIRS_PER_TASK,
        "all_task_metrics_present": (not e1_present) or all(task in metrics["e1"]["metrics"][method] for method in METHODS for task in TASKS),
        "identifiability_pass": (not e1_present) or metrics["e1"]["identifiability"]["pixel_equal_count"] == 0,
        "no_training": metrics.get("training_performed") is False and metrics.get("learned_classifier") is False,
        "downstream_closed": not any(metrics.get(key, True) for key in ("E2_opened", "conjugacy_opened", "confirm_created", "cracks_accessed", "expert_accessed")),
        "report_exists": (RESULT_ROOT / "ANZA_EK_E0_E1_REPORT.md").exists(),
        "figures_exist": all(Path(path).exists() for path in metrics.get("figures", [])),
    }
    passed = all(checks.values())
    result = {"validator_status": "PASS" if passed else "FAIL", "research_status": metrics.get("status"), "checks": checks}
    _write(RESULT_ROOT / "validator.json", result)
    if not passed:
        raise ValueError("ANZA-EK E0/E1 final validation failed")
    return result
