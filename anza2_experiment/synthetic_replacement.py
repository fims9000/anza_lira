"""Independent Phase-2B replacement after the Phase-2A path endpoint saturated.

Phase-2A remains immutable.  This protocol tests the already pre-specified
junction/branch-preservation question on a new seed stream while requiring path
and false-bridge non-inferiority.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .synthetic_mechanism import (
    METHODS,
    SAMPLES_PER_STRATUM,
    _branch_rows,
    _canonical_hash,
    _path_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase2b"
PHASE2A_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase2"
PHASE2A_METRICS_SHA256 = "04b35a97c830b682f682084498673daf280e1c81dad407e850be199e8e15e383"
REPLACEMENT_CONFIRM_SEED_BASE = 630_000_000
BOOTSTRAP_RESAMPLES = 10_000


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "anza2_phase2b_independent_branch_replacement_v1",
        "reason": "Phase-2A path TPR saturated at 1.0 for Legacy and ANZA-2; its FAIL is frozen and not reinterpreted.",
        "phase2a_metrics_sha256": PHASE2A_METRICS_SHA256,
        "development_threshold_source": "frozen Phase-2A development threshold_freeze.json",
        "replacement_confirm_seed_base": REPLACEMENT_CONFIRM_SEED_BASE,
        "samples_per_stratum": SAMPLES_PER_STRATUM,
        "primary_metric": "junction branch recall delta: ANZA2 absolute minus legacy normalized",
        "minimum_branch_recall_delta": 0.08,
        "branch_delta_ci_lower_required": 0.0,
        "path_tpr_noninferiority_margin": -0.02,
        "false_bridge_noninferiority_margin": 0.01,
        "anza_branch_recall_min_each_stratum": 0.95,
        "bootstrap_unit": "paired synthetic index across X/T/Y strata",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confirm_opened_before_protocol_freeze": False,
        "training_performed": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }


def _path_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    result = {}
    for method in METHODS:
        predicted = np.asarray([row[method] >= thresholds[method] for row in rows])
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        positives = int(np.sum(labels == 1)); negatives = int(np.sum(labels == 0))
        result[method] = {
            "tpr": tp / positives, "fpr": fp / negatives,
            "tp": tp, "fp": fp, "positives": positives, "negatives": negatives,
        }
    return result


def _branch_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    result = {}
    for method in METHODS:
        values = np.asarray([row[method] for row in rows])
        hits = values >= thresholds[method]
        by_case = {}
        for case in sorted({row["case"] for row in rows}):
            indices = np.asarray([row["case"] == case for row in rows])
            by_case[case] = {
                "recall": float(np.mean(hits[indices])),
                "hits": int(np.sum(hits[indices])),
                "total": int(np.sum(indices)),
            }
        result[method] = {
            "recall": float(np.mean(hits)),
            "hits": int(np.sum(hits)),
            "total": int(hits.size),
            "by_case": by_case,
        }
    return result


def _paired_bootstrap(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> tuple[float, list[float]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row["index"]), []).append(row)
    keys = sorted(by_index)
    per_index = []
    for key in keys:
        group = by_index[key]
        anza = np.mean([row["anza2_absolute"] >= thresholds["anza2_absolute"] for row in group])
        legacy = np.mean([row["legacy_global_normalized"] >= thresholds["legacy_global_normalized"] for row in group])
        per_index.append(float(anza - legacy))
    rng = np.random.default_rng(20260818)
    samples = [float(np.mean(rng.choice(per_index, size=len(per_index), replace=True))) for _ in range(BOOTSTRAP_RESAMPLES)]
    return float(np.mean(per_index)), [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def run_phase2b(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if _digest(PHASE2A_ROOT / "metrics.json") != PHASE2A_METRICS_SHA256:
        raise ValueError("frozen Phase-2A metrics changed")
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    protocol_hash = _canonical_hash(protocol)
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (output_root / "protocol_hash.txt").write_text(protocol_hash + "\n")
    threshold_freeze = json.loads((PHASE2A_ROOT / "threshold_freeze.json").read_text())
    thresholds = {key: float(value) for key, value in threshold_freeze["thresholds"].items()}
    open_receipt = {
        "status": "REPLACEMENT_CONFIRM_OPEN_AUTHORIZED",
        "protocol_sha256": protocol_hash,
        "phase2a_metrics_sha256": PHASE2A_METRICS_SHA256,
        "threshold_freeze_sha256": _digest(PHASE2A_ROOT / "threshold_freeze.json"),
        "replacement_confirm_rows_opened": 0,
        "training_performed": False,
        "expert_data_accessed": False,
    }
    (output_root / "open_receipt.json").write_text(json.dumps(open_receipt, indent=2, sort_keys=True) + "\n")
    paths = _path_rows("confirm", seed_base=REPLACEMENT_CONFIRM_SEED_BASE)
    branches = _branch_rows("confirm", seed_base=REPLACEMENT_CONFIRM_SEED_BASE)
    path_metrics = _path_metrics(paths, thresholds)
    branch_metrics = _branch_metrics(branches, thresholds)
    delta, ci = _paired_bootstrap(branches, thresholds)
    anza_path = path_metrics["anza2_absolute"]
    legacy_path = path_metrics["legacy_global_normalized"]
    anza_branch = branch_metrics["anza2_absolute"]
    gate = bool(
        delta >= protocol["minimum_branch_recall_delta"]
        and ci[0] > protocol["branch_delta_ci_lower_required"]
        and anza_path["tpr"] - legacy_path["tpr"] >= protocol["path_tpr_noninferiority_margin"]
        and anza_path["fpr"] - legacy_path["fpr"] <= protocol["false_bridge_noninferiority_margin"]
        and min(row["recall"] for row in anza_branch["by_case"].values()) >= protocol["anza_branch_recall_min_each_stratum"]
    )
    result = {
        "status": "PHASE2_GEOMETRY_SELECTIVITY_PASS" if gate else "STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE",
        "protocol_sha256": protocol_hash,
        "phase2a_status_preserved": "STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE",
        "path_metrics": path_metrics,
        "branch_metrics": branch_metrics,
        "anza_minus_legacy_branch_recall": delta,
        "anza_minus_legacy_branch_recall_ci95": ci,
        "phase2b_gate_pass": gate,
        "training_performed": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
        "claim_boundary": "Independent oracle-field replacement confirms branch selectivity only; learned image inference remains untested.",
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_root / "bootstrap.json").write_text(json.dumps({
        "unit": protocol["bootstrap_unit"], "resamples": BOOTSTRAP_RESAMPLES,
        "delta": delta, "ci95": ci,
    }, indent=2, sort_keys=True) + "\n")
    for filename, rows in (("per_path.csv", paths), ("per_branch.csv", branches)):
        with (output_root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output_root / "PHASE2B_REPORT.md").write_text(f"""# ANZA-2 Phase 2B replacement report

Phase-2A remains frozen as FAIL because its primary path endpoint saturated for both Legacy and ANZA-2. This independent seed stream used the already frozen thresholds and a predeclared branch-selectivity primary metric.

- Status: `{result['status']}`
- ANZA-2 branch recall: `{anza_branch['recall']:.6f}`
- Legacy branch recall: `{branch_metrics['legacy_global_normalized']['recall']:.6f}`
- Paired delta: `{delta:.6f}`, 95% CI `{ci}`
- ANZA-2 path TPR/FPR: `{anza_path['tpr']:.6f}` / `{anza_path['fpr']:.6f}`
- Legacy path TPR/FPR: `{legacy_path['tpr']:.6f}` / `{legacy_path['fpr']:.6f}`
- Training: no
- CRACKS/expert: not accessed

This is controlled oracle-field mechanism evidence, not a learned or real-data result.
""")
    return result
