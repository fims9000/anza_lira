"""Independent RC1 artifact and decision validator."""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from .protocol import canonical_hash, write_json
from .rc1_protocol import ROOT, VARIANTS, load_frozen, verify_parent_immutable


VALID_STATUSES = {
    "ANZA_RC1_HIGH_PRECISION_LOW_LABEL_PASS",
    "STOP_ANZA_LOW_LABEL_GAIN_WAS_OPERATING_POINT_SPECIFIC",
    "ANZA_PRIOR_RELATIVE_ONLY_NOT_PRACTICAL",
    "ANZA_RC1_STRUCTURAL_SIGNAL_SAFETY_FAIL",
    "STOP_ANZA_RC1_HIGH_PRECISION_INFEASIBLE",
}


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else (0.0 if a == 0 else math.inf)


def validate_rc1() -> dict:
    protocol, split = load_frozen()
    common = [ROOT / "protocol.json", ROOT / "protocol_hash.txt", ROOT / "split_manifest.json", ROOT / "label_budget.json",
              ROOT / "parent_freeze.json", ROOT / "threshold_freeze.json", ROOT / "metrics.json", ROOT / "ANZA_LEADS_RC1_REPORT.md"]
    checks = {f"exists:{path.name}": path.is_file() for path in common}
    if not all(checks.values()):
        result = {"status": "FAIL", "checks": checks}; write_json(ROOT / "validator.json", result); return result
    freeze = json.loads((ROOT / "threshold_freeze.json").read_text()); freeze_sha = freeze.pop("freeze_sha256")
    metrics = json.loads((ROOT / "metrics.json").read_text())
    checks.update({
        "parent_immutable": verify_parent_immutable(), "parent_status_preserved": protocol["parent_status_immutable"] == "STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL",
        "freeze_hash": freeze_sha == canonical_hash(freeze), "expert_locked": metrics.get("expert_data_accessed") is False,
        "downstream_locked": all(metrics.get(f"{key}_opened") is False for key in ("seeds_42_43", "anza_ms", "ssl", "domain_shift", "lira")),
        "valid_status": metrics.get("status") in VALID_STATUSES,
        "fresh_evaluation": not bool((set(split["calibration"]) | set(split["development"])) & (set(split["old_a1_active_sections_excluded_from_rc1_evaluation"]) | set(split["old_a1_selection_sections_excluded_from_rc1"]))),
    })
    if freeze["status"] == "HIGH_PRECISION_INFEASIBLE":
        checks.update({"correct_infeasible_status": metrics["status"] == "STOP_ANZA_RC1_HIGH_PRECISION_INFEASIBLE", "development_closed": metrics.get("development_opened") is False})
    else:
        required = [ROOT / "development_open_receipt.json", ROOT / "development_per_section.csv", ROOT / "development_per_annotator.csv",
                    ROOT / "development_frontiers.csv", ROOT / "frontier_summary.json", ROOT / "bootstrap.json"]
        checks.update({f"exists:{path.name}": path.is_file() for path in required})
        if all(path.is_file() for path in required):
            with (ROOT / "development_per_section.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in VARIANTS}
            keys = ("precision", "dice", "cldice", "auprc", "unsupported_white_foreground_fraction")
            summary = {variant: {key: float(np.mean([float(row[key]) for row in local])) for key in keys} for variant, local in by_variant.items()}
            l0, l2, l3 = (summary[key] for key in VARIANTS)
            bootstrap = json.loads((ROOT / "bootstrap.json").read_text())
            gate = protocol["gate"]
            recomputed = {
                "development_precision_L2": l2["precision"] >= protocol["development_precision_min"],
                "development_precision_L3": l3["precision"] >= protocol["development_precision_min"],
                "cldice_gain": l3["cldice"] - l2["cldice"] >= gate["cldice_delta_min"],
                "cldice_ci_positive": bootstrap["cldice"]["ci95_low"] > 0,
                "dice_noninferior_L2": l3["dice"] - l2["dice"] >= gate["dice_delta_min"],
                "cldice_noninferior_backbone": l3["cldice"] - l0["cldice"] >= gate["backbone_cldice_delta_min"],
                "dice_noninferior_backbone": l3["dice"] - l0["dice"] >= gate["backbone_dice_delta_min"],
                "auprc_noninferior": l3["auprc"] - l2["auprc"] >= gate["auprc_delta_min"],
                "unsupported_white_vs_L2": _ratio(l3["unsupported_white_foreground_fraction"], l2["unsupported_white_foreground_fraction"]) <= gate["unsupported_white_ratio_max"],
                "unsupported_white_vs_L0": _ratio(l3["unsupported_white_foreground_fraction"], l0["unsupported_white_foreground_fraction"]) <= gate["unsupported_white_ratio_max"],
            }
            checks.update({"all_variants_present": all(len(local) == len(split["development"]) for local in by_variant.values()),
                           "metrics_recomputed": recomputed == metrics["checks"], "gate_recomputed": bool(metrics["gate_pass"]) == all(recomputed.values())})
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "research_status": metrics.get("status"), "checks": checks,
              "expert_data_accessed": False}
    write_json(ROOT / "validator.json", result)
    return result
