#!/usr/bin/env python3
"""Independent lock and artifact validator for final LIRA F0--F3 slice."""

from __future__ import annotations

import json
from pathlib import Path

from lira_final.data.splits import build_split_manifest
from lira_final.io import write_json
from lira_final.protocol import RESULT_ROOT, ROOT, protocol_hash


def validate() -> dict[str, object]:
    failures = []
    f0 = json.loads((RESULT_ROOT / "f0_freeze/historical_registry.json").read_text())
    f1 = json.loads((RESULT_ROOT / "f1_gap_audit/metrics.json").read_text())
    splits = json.loads((RESULT_ROOT / "f1_gap_audit/split_manifest.json").read_text())
    if f0.get("status") != "F0_PASS" or f0.get("old_results_modified") is not False:
        failures.append("F0 immutable registry invalid")
    if f0.get("protocol_sha256") != protocol_hash() or f1.get("protocol_sha256") != protocol_hash():
        failures.append("protocol hash drift")
    if splits != build_split_manifest():
        failures.append("split manifest drift")
    split_sets = {name: set(values) for name, values in splits["splits"].items()}
    names = list(split_sets)
    if any(split_sets[names[i]] & split_sets[names[j]] for i in range(len(names)) for j in range(i + 1, len(names))):
        failures.append("section overlap")
    if "expert" in splits.get("heldout_annotators", []):
        failures.append("expert annotator entered splits")
    if f1.get("expert_accessed") is not False or f1.get("confirm", {}).get("inference_opened") is not False:
        failures.append("expert or confirm lock failed")
    cache = RESULT_ROOT / "f1_gap_audit/dense_cache"
    confirm_cache = [cache / f"section_{section:03d}.npy" for section in splits["splits"]["lira_confirm"]]
    if any(path.exists() for path in confirm_cache):
        failures.append("confirm dense inference was opened")
    dev = f1.get("splits", {}).get("lira_development", {})
    if f1.get("status") == "STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT" and int(dev.get("positive_gaps", 0)) >= 75:
        failures.append("data insufficiency STOP inconsistent with absolute floor")
    locked_text = (RESULT_ROOT / "f3_relation_s41/LIRA_REAL_RELATION_S41_REPORT.md").read_text()
    if f1.get("status", "").startswith("STOP_") and "LOCKED_NOT_RUN" not in locked_text:
        failures.append("F3 was not locked after F1 STOP")
    result = {
        "validator_status": "PASS" if not failures else "FAIL",
        "research_status": f1.get("status"),
        "failures": failures,
        "expert_accessed": False,
        "confirm_opened": False,
        "f2_opened": False,
        "f3_opened": False,
    }
    write_json(RESULT_ROOT / "f1_gap_audit/validator.json", result)
    return result


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["validator_status"] == "PASS" else 1)

