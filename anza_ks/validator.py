"""Pre-run and final ANZA-KS K0/K1 validation."""

from __future__ import annotations

import csv
import json
from typing import Any

from .benchmark.matched_generator import SPLIT_SIZES, TASKS
from .features import METHODS
from .protocol import FREEZE_ROOT, RESULT_ROOT, canonical_hash
from .runner import freeze_k0_inputs, source_manifest


def _json(path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_pre_run() -> dict[str, Any]:
    protocol, benchmark, receipt = freeze_k0_inputs()
    checks = {
        "protocol_hash_matches": receipt["protocol_sha256"] == canonical_hash(protocol),
        "benchmark_hash_matches": receipt["benchmark_sha256"] == canonical_hash(benchmark),
        "source_hash_matches": receipt["source_sha256"] == source_manifest()["sha256"],
        "k0_math_pass": json.loads((FREEZE_ROOT / "k0_math.json").read_text())["status"] == "ANZA_KS_K0_MATH_PASS",
        "five_tasks_frozen": len(TASKS) == 5,
        "sample_sizes_frozen": SPLIT_SIZES == {"train": 2048, "dev": 1024, "confirm": 2048},
        "static_gate_pass": benchmark["static_match_status"] == "ANZA_KS_STATIC_MATCH_PASS",
        "all_static_aurocs_valid": all(0.45 <= row["static_dev_auroc"] <= 0.60 for row in benchmark["static_diagnostics"]),
        "static_tolerance_pass": all(row["maximum_static_pair_delta"] <= 1e-8 for row in benchmark["static_diagnostics"]),
        "confirm_hashed_and_locked": len(benchmark["confirm_stream_sha256"]) == 64 and benchmark["confirm_samples_exposed"] is False,
        "symbolic_scoring_not_started": receipt["symbolic_scores_computed"] is False and not (RESULT_ROOT / "per_pair.csv").exists(),
        "four_capacity_controls": tuple(protocol["methods"]) == METHODS,
        "downstream_locked": not any(protocol[key] for key in ("K2_opened", "confirm_evaluated", "cracks_accessed", "expert_accessed")),
    }
    passed = all(checks.values())
    result = {
        "validator_status": "PASS" if passed else "FAIL",
        "research_status": "ANZA_KS_K0_PRE_RUN_PASS" if passed else "ANZA_KS_K0_PRE_RUN_FAIL",
        "run_authorized": passed,
        "checks": checks,
        "protocol_sha256": receipt["protocol_sha256"],
        "benchmark_sha256": receipt["benchmark_sha256"],
        "source_sha256": receipt["source_sha256"],
    }
    _json(FREEZE_ROOT / "pre_run_validator.json", result)
    if not passed:
        raise ValueError("ANZA-KS K0 pre-run validation failed")
    return result


def validate_final() -> dict[str, Any]:
    metrics = json.loads((RESULT_ROOT / "metrics.json").read_text())
    protocol = json.loads((FREEZE_ROOT / "protocol.json").read_text())
    benchmark = json.loads((FREEZE_ROOT / "benchmark_manifest.json").read_text())
    receipt = json.loads((FREEZE_ROOT / "freeze_receipt.json").read_text())
    with (RESULT_ROOT / "per_pair.csv").open(newline="") as handle:
        pair_rows = sum(1 for _ in csv.DictReader(handle))
    with (RESULT_ROOT / "per_task.csv").open(newline="") as handle:
        task_rows = sum(1 for _ in csv.DictReader(handle))
    recognized = {
        "STOP_STATIC_MATCH_BENCH_INVALID",
        "STOP_SYMBOLIC_DYNAMICS_NO_INCREMENTAL_SIGNAL",
        "STOP_KOLMOGOROV_FEATURES_REDUNDANT",
        "STOP_ANOSOV_NOT_SPECIFIC_SHEAR_EQUAL",
        "ANZA_KS_CAUSAL_FEATURE_PASS",
    }
    checks = {
        "protocol_unchanged": receipt["protocol_sha256"] == canonical_hash(protocol) == metrics["protocol_sha256"],
        "benchmark_unchanged": receipt["benchmark_sha256"] == canonical_hash(benchmark) == metrics["benchmark_sha256"],
        "source_unchanged": receipt["source_sha256"] == source_manifest()["sha256"] == metrics["source_sha256"],
        "recognized_status": metrics["status"] in recognized,
        "all_pair_rows": pair_rows == len(METHODS) * len(TASKS) * SPLIT_SIZES["dev"],
        "all_task_rows": task_rows == len(METHODS) * len(TASKS),
        "all_metrics": all(task in metrics["k1"]["metrics"][method] for method in METHODS for task in TASKS),
        "static_gate_still_passes": metrics["static_match_status"] == "ANZA_KS_STATIC_MATCH_PASS",
        "tiny_readout_only": metrics["tiny_logistic_readouts_trained"] is True and metrics["segmentation_training_performed"] is False,
        "confirm_closed": metrics["confirm_evaluated"] is False,
        "downstream_closed": not any(metrics[key] for key in ("K2_opened", "cracks_accessed", "expert_accessed")),
        "required_artifacts": all(
            (RESULT_ROOT / name).exists()
            for name in (
                "protocol.json",
                "protocol_hash.txt",
                "benchmark_manifest.json",
                "static_match_diagnostics.csv",
                "per_pair.csv",
                "per_task.csv",
                "feature_dimensions.json",
                "operating_curves.csv",
                "bootstrap.json",
                "metrics.json",
                "ANZA_KS_K0_K1_REPORT.md",
            )
        ),
    }
    passed = all(checks.values())
    result = {"validator_status": "PASS" if passed else "FAIL", "research_status": metrics["status"], "checks": checks}
    _json(RESULT_ROOT / "validator.json", result)
    if not passed:
        raise ValueError("ANZA-KS K0/K1 final validation failed")
    return result
