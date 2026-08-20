"""Fail-closed validation for Candidate Audit V2."""

from __future__ import annotations

import csv
import json
from typing import Any

from .candidate_audit_v2 import AUDIT_PROTOCOL, DENSE_CHECKPOINT, PARENT, RESULT, _sha, _source_manifest, audit_protocol_hash


def validate() -> dict[str, Any]:
    required = ["protocol.json", "protocol_hash.txt", "source_manifest.json", "per_case.csv", "miss_taxonomy.csv", "endpoint_errors.csv", "recall_vs_k.csv", "taxonomy.json", "taxonomy_by_scene.csv", "gap_mismatch.json", "implementation_audit.json", "zero_training_receipt.json", "metrics.json", "CANDIDATE_AUDIT_V2_REPORT.md"]
    missing = [name for name in required if not (RESULT / name).is_file()]
    if missing: raise ValueError(f"missing Candidate Audit V2 artifacts: {missing}")
    metrics = json.loads((RESULT / "metrics.json").read_text()); receipt = json.loads((RESULT / "zero_training_receipt.json").read_text()); taxonomy = json.loads((RESULT / "taxonomy.json").read_text()); source = json.loads((RESULT / "source_manifest.json").read_text()); parent = json.loads((PARENT / "metrics.json").read_text())
    misses = list(csv.DictReader((RESULT / "miss_taxonomy.csv").open())); per_case = list(csv.DictReader((RESULT / "per_case.csv").open()))
    checks = {
        "protocol_hash": metrics["protocol_sha256"] == audit_protocol_hash() == (RESULT / "protocol_hash.txt").read_text().strip(),
        "source_hash": metrics["source_sha256"] == source["sha256"] == _source_manifest()["sha256"],
        "checkpoint_immutable": receipt["checkpoint_before"] == receipt["checkpoint_after"] == _sha(DENSE_CHECKPOINT),
        "parent_reproduced": metrics["parent_miss_set_exact"] and metrics["v1_misses"] == 206 and abs(metrics["v1_recall_at_6px"] - parent["candidate_recall"]["candidate_recall"]) < 1e-12,
        "distance_bins_reproduced": metrics["v1_distance_bins"] == {"le_6": 818, "gt_6_le_8": 120, "gt_8_le_10": 56, "gt_10_or_missing": 30},
        "full_top8_reproduced": metrics["misses_with_full_top8"] == 143,
        "sample_size": len(per_case) == 2048 and metrics["positive_sources"] == 1024,
        "taxonomy_exhaustive": len(misses) == 206 and taxonomy["total"] == 206 and sum(taxonomy["counts"].values()) == 206 and all(row["category"] in AUDIT_PROTOCOL["taxonomy_priority"] for row in misses),
        "k_curve": {int(row["k"]) for row in csv.DictReader((RESULT / "recall_vs_k.csv").open())} == {4, 8, 12, 16, 24, 32},
        "zero_training": not receipt["training_opened"] and not receipt["optimizer_created"],
        "locks": not receipt["confirm_opened"] and not receipt["cracks_accessed"] and not receipt["expert_accessed"],
        "status": metrics["status"] == "CANDIDATE_AUDIT_V2_COMPLETE",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError(f"Candidate Audit V2 validation failed: {failed}")
    return {"validator": "PASS", "research_status": metrics["status"], "checks": checks, "training_opened": False, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
