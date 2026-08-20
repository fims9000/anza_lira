"""Fail-closed validation for TRACEGRAPH SBPP V3-A artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from anza_tracegraph.frozen_source import DENSE_CHECKPOINT
from .runner import _manifest


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/anza_tracegraph/sbpp_v3_a"


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict[str, Any]:
    required = ("protocol.json", "protocol_hash.txt", "source_manifest.json", "generator_validation.json", "split_manifest.json", "calibration_sweep.csv", "sbpp_freeze.json", "development_per_case.csv", "recall_vs_k.csv", "per_stratum.csv", "miss_taxonomy.csv", "metrics.json", "zero_training_receipt.json", "TRACEGRAPH_SBPP_V3_A_REPORT.md")
    missing = [name for name in required if not (RESULT / name).is_file()]
    if missing: raise ValueError(f"missing SBPP V3-A artifacts: {missing}")
    protocol = json.loads((RESULT / "protocol.json").read_text()); metrics = json.loads((RESULT / "metrics.json").read_text()); freeze = json.loads((RESULT / "sbpp_freeze.json").read_text()); receipt = json.loads((RESULT / "zero_training_receipt.json").read_text()); generator = json.loads((RESULT / "generator_validation.json").read_text()); split = json.loads((RESULT / "split_manifest.json").read_text()); source = json.loads((RESULT / "source_manifest.json").read_text())
    rows = list(csv.DictReader((RESULT / "development_per_case.csv").open())); misses = list(csv.DictReader((RESULT / "miss_taxonomy.csv").open())); strata = list(csv.DictReader((RESULT / "per_stratum.csv").open()))
    calibration_stop = metrics["status"] == "STOP_SBPP_CALIBRATION_COVERAGE_FAIL"
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    positive_rows = [row for row in rows if row["positive"] == "1"]
    checks = {
        "protocol_hash": protocol_hash == (RESULT / "protocol_hash.txt").read_text().strip() == metrics["protocol_sha256"],
        "source_hash": source["sha256"] == _manifest()["sha256"],
        "generator_valid": generator["validator"] == "PASS" and all(generator["checks"].values()),
        "confirm_hash_only": split["confirm"]["hash_only"] and not split["confirm"]["inference_opened"] and not split["confirm"]["metrics_opened"],
        "checkpoint_immutable": receipt["checkpoint_before"] == receipt["checkpoint_after"] == _sha(DENSE_CHECKPOINT),
        "calibration_only_selection": freeze["selection_split"] == "calibration" and freeze["tau_micro"] in protocol["tau_micro_candidates"],
        "development_sample_size": calibration_stop and not rows or len(positive_rows) >= 2048 and len(rows) - len(positive_rows) >= 1024,
        "taxonomy_exhaustive": calibration_stop and not misses or len(misses) == sum(row["branch_recalled_at_12"] == "0" for row in positive_rows) and all(row["miss_category"] in {"A3", "B3", "C3", "D3", "E3", "F3"} for row in misses),
        "no_f3": calibration_stop or metrics["taxonomy"].get("F3", 0) == 0,
        "curve": calibration_stop or {int(row["k"]) for row in csv.DictReader((RESULT / "recall_vs_k.csv").open())} == {4, 8, 12, 16},
        "strata_sample_floor": calibration_stop or all(int(row["positive_sources"]) >= 128 for row in strata),
        "zero_training": not receipt["training_opened"] and not receipt["optimizer_created"] and not receipt["p0_opened"] and not receipt["transformer_opened"] and not receipt["path_opened"],
        "downstream_locks": not receipt["confirm_evaluated"] and not receipt["cracks_accessed"] and not receipt["expert_accessed"],
        "status_consistent": metrics["status"] in {"SBPP_BRANCH_COVERAGE_PASS", "STOP_SBPP_BRANCH_COVERAGE_FAIL", "STOP_SBPP_CANDIDATE_BUDGET_FAIL", "STOP_TRACEGRAPH_V2_GENERATOR_INVALID", "STOP_SBPP_CALIBRATION_COVERAGE_FAIL"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError(f"SBPP V3-A validation failed: {failed}")
    return {"validator": "PASS", "research_status": metrics["status"], "checks": checks, "training_opened": False, "confirm_evaluated": False, "cracks_accessed": False, "expert_accessed": False}
