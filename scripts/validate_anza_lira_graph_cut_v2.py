#!/usr/bin/env python3
"""Independent artifact/lock validator for Graph-Cut Intervention V2."""

from __future__ import annotations

from collections import Counter
import csv
import json

from lira_graph_cut_v2.benchmark import split_manifest
from lira_graph_cut_v2.protocol import PROTOCOL, RESULT_ROOT, ROOT, protocol_hash


def validate() -> dict[str, object]:
    failures = []
    freeze = json.loads((RESULT_ROOT / "freeze/freeze_receipt.json").read_text())
    retention = json.loads((RESULT_ROOT / "benchmark/retention.json").read_text())
    manifest = split_manifest()
    if freeze.get("protocol_sha256") != protocol_hash() or retention.get("protocol_sha256") != protocol_hash():
        failures.append("protocol hash drift")
    if freeze.get("parent_status") != "STOP_LIRA_INTERVENTION_CANDIDATE" or retention.get("parent_stop_changed") is not False:
        failures.append("parent V1 STOP changed")
    if freeze.get("split_manifest") != manifest:
        failures.append("split manifest drift")
    split_sets = [set(value) for value in manifest["splits"].values()]
    if any(split_sets[i] & split_sets[j] for i in range(len(split_sets)) for j in range(i + 1, len(split_sets))):
        failures.append("section overlap")
    recomputed = {}
    for split in ("gc_calibration", "gc_development"):
        with (RESULT_ROOT / f"benchmark/{split}_eligibility.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        eligible = sum(int(row["eligible_before_treatment"]) for row in rows)
        valid = sum(row["status"] == "VALID" for row in rows)
        ratio = valid / eligible if eligible else 0.0
        statuses = Counter(row["status"] for row in rows)
        stored = retention["splits"][split]
        if eligible != int(stored["eligible_before_treatment"]) or valid != int(stored["valid_cases"]) or abs(ratio - float(stored["retention"])) > 1e-12:
            failures.append(f"retention drift: {split}")
        for row in rows:
            radius = row.get("radius_px", "")
            if radius and int(radius) not in PROTOCOL["treatment"]["candidate_radii_px"]:
                failures.append(f"unfrozen radius: {radius}")
                break
            if row["status"] == "VALID" and row.get("post_connected") != "0":
                failures.append(f"accepted connected treatment: {split}")
                break
            if row["status"] == "INVALID_COLLATERAL_TRACE" and float(row["collateral_fraction"]) <= float(PROTOCOL["treatment"]["maximum_collateral_fraction"]):
                failures.append(f"collateral status mismatch: {split}")
                break
        recomputed[split] = {"eligible": eligible, "valid": valid, "retention": ratio, "statuses": dict(statuses)}
    if retention.get("status") != "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE" or any(value["retention"] >= float(PROTOCOL["treatment"]["minimum_retention"]) for value in recomputed.values()):
        failures.append("benchmark STOP inconsistent with retention gate")
    candidate = json.loads((RESULT_ROOT / "candidate/validator.json").read_text())
    if candidate.get("candidate_opened") is not False or (RESULT_ROOT / "candidate/per_case.csv").exists():
        failures.append("SBPP candidate stage opened after benchmark STOP")
    if (RESULT_ROOT / "relation/checkpoint.pt").exists() or (RESULT_ROOT / "path").exists():
        failures.append("P0/path opened")
    confirm_cache = ROOT / "results/lira_final/f1_gap_audit/dense_cache"
    if any((confirm_cache / f"section_{section:03d}.npy").exists() for section in manifest["splits"]["gc_confirm"]):
        failures.append("confirm dense inference opened")
    figures = json.loads((RESULT_ROOT / "benchmark/figures/figure_manifest.json").read_text())
    if len(figures.get("artifacts", [])) != 6 or figures.get("candidate_opened") is not False:
        failures.append("figure manifest incomplete")
    result = {
        "validator_status": "PASS" if not failures else "FAIL",
        "research_status": retention.get("status"),
        "failures": failures,
        "recomputed": recomputed,
        "candidate_opened": False,
        "p0_opened": False,
        "path_opened": False,
        "confirm_opened": False,
        "expert_accessed": False,
    }
    (RESULT_ROOT / "benchmark/validator.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["validator_status"] == "PASS" else 1)
