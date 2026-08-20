#!/usr/bin/env python3
"""Validate the bounded ANZA-KS K1 evaluator audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "anza_ks" / "k0_k1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    audit = json.loads((RESULT_ROOT / "gate_audit_r1.json").read_text())
    checks = {
        "audit_version": audit["audit_version"] == "ANZA_KS_K1_GATE_AUDIT_R1",
        "immutable_metrics": audit["inputs"]["metrics_sha256"] == sha256(RESULT_ROOT / "metrics.json"),
        "immutable_rows": audit["inputs"]["per_pair_sha256"] == sha256(RESULT_ROOT / "per_pair.csv"),
        "no_retraining": audit["tiny_readouts_retrained"] is False and audit["features_recomputed"] is False,
        "benchmark_unchanged": audit["benchmark_changed"] is False,
        "downstream_closed": not any(audit[key] for key in ("confirm_evaluated", "K2_opened", "cracks_accessed", "expert_accessed")),
        "report_exists": (RESULT_ROOT / "ANZA_KS_K0_K1_GATE_AUDIT_R1.md").exists(),
    }
    result = {"validator_status": "PASS" if all(checks.values()) else "FAIL", "research_status": audit["status"], "checks": checks}
    (RESULT_ROOT / "gate_audit_r1_validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise SystemExit("ANZA-KS gate audit validation failed")
    print(audit["status"])
