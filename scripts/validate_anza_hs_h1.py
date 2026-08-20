#!/usr/bin/env python3
"""Fail-closed validator for bounded ANZA-HS H1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]; H0 = ROOT / "results" / "anza_hs" / "h0"; H1 = ROOT / "results" / "anza_hs" / "h1"


def _load(path: Path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate() -> dict:
    required = ("parent_h0.json", "data_access_log.json", "cuda_smoke.json", "threshold_freeze.json", "metrics.json", "calibration_curves.json", "operator_diagnostics.json", "raw_per_sample.csv", "ANZA_HS_H1_REPORT.md", "TASK_STATE.json")
    failures = [name for name in required if not (H1 / name).is_file() or not (H1 / name).stat().st_size]
    if failures:
        return {"status": "FAIL", "research_status": "ANZA_HS_H1_INVALID", "failures": failures}
    protocol = _load(H0 / "protocol.json"); metrics = _load(H1 / "metrics.json"); access = _load(H1 / "data_access_log.json")
    if _load(H0 / "validator.json").get("research_status") != "ANZA_HS_H0_PASS" or metrics.get("protocol_sha256") != _hash(protocol):
        failures.append("H0/protocol provenance mismatch")
    expected = {"B0_backbone", "B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic"}
    if set(metrics.get("variants", {})) != expected:
        failures.append("B0-B3 matrix incomplete")
    for variant, value in metrics.get("variants", {}).items():
        run = value.get("run", {})
        if run.get("status") != "COMPLETE" or run.get("epoch") != 20 or run.get("protocol_sha256") != metrics.get("protocol_sha256"):
            failures.append(f"incomplete run: {variant}")
        if not Path(run.get("checkpoint", "missing")).is_file(): failures.append(f"missing checkpoint: {variant}")
    comparison = metrics.get("comparison", {}); checks = comparison.get("gate_checks", {})
    computed = bool(checks.get("dice_noninferiority") and (checks.get("cldice_gain") or checks.get("fragmentation_reduction")))
    expected_status = "ANZA_HS_H1_PASS" if computed else "HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL"
    if metrics.get("gate_pass") is not computed or metrics.get("status") != expected_status:
        failures.append("H1 status does not follow frozen gate")
    with (H1 / "raw_per_sample.csv").open(newline="") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != 4 * 220 or {row["variant"] for row in rows} != expected or {int(row["index"]) for row in rows} != set(range(44, 264)):
        failures.append("dev gate sample contract failed")
    if any(not math.isfinite(float(row[key])) for row in rows for key in ("dice", "precision", "recall", "cldice", "fragmentation")):
        failures.append("non-finite raw metric")
    expected_access = {"synthetic_train": True, "synthetic_dev_calibration": "dev[0:44]", "synthetic_dev_gate": "dev[44:264]", "confirm": False, "test": False, "cracks": False, "continuation": False, "expert": False}
    if access != expected_access: failures.append("data access lock drift")
    for key in ("confirm_opened", "test_opened", "H2_opened", "cracks_accessed", "continuation_trained", "expert_accessed", "lambda_tuned", "M_tuned", "base_scale_alternative_used"):
        if metrics.get(key) is not False: failures.append(f"forbidden downstream/tuning flag: {key}")
    result = {"status": "PASS" if not failures else "FAIL", "research_status": expected_status if not failures else "ANZA_HS_H1_INVALID",
              "failures": failures, "h1_gate_pass": computed and not failures, "confirm_opened": False, "H2_opened": False,
              "cracks_accessed": False, "continuation_trained": False, "expert_accessed": False,
              "next_action": "Freeze H2 separately" if computed and not failures else "STOP; do not advance ANZA-HS local hyperbolic claim"}
    (H1 / "validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    value = validate(); print(json.dumps(value, indent=2, sort_keys=True)); raise SystemExit(0 if value["status"] == "PASS" else 1)
