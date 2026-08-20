"""Independent artifact validators for LEADS A0 and A1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .model import LEADS_VARIANTS
from .protocol import A0_ROOT, A1_ROOT, PROTOCOL, canonical_hash, protocol_hash, write_json


def validate_a0() -> dict[str, Any]:
    required = [
        A0_ROOT / "protocol.json", A0_ROOT / "protocol_hash.txt", A0_ROOT / "split_manifest.json",
        A0_ROOT / "label_subset_manifest.json", A0_ROOT / "data_access_log.json", A0_ROOT / "validator.json",
    ]
    checks = {f"exists:{path.name}": path.is_file() for path in required}
    if all(checks.values()):
        protocol = json.loads((A0_ROOT / "protocol.json").read_text())
        split = json.loads((A0_ROOT / "split_manifest.json").read_text())
        subsets = json.loads((A0_ROOT / "label_subset_manifest.json").read_text())
        audit = json.loads((A0_ROOT / "validator.json").read_text())
        access = json.loads((A0_ROOT / "data_access_log.json").read_text())
        checks.update({
            "protocol_hash": canonical_hash(protocol) == protocol_hash() == (A0_ROOT / "protocol_hash.txt").read_text().strip(),
            "split_hash": split["sha256"] == canonical_hash({key: value for key, value in split.items() if key != "sha256"}),
            "subset_hash": subsets["sha256"] == canonical_hash({key: value for key, value in subsets.items() if key != "sha256"}),
            "audit_pass": audit.get("status") == "ANZA_LEADS_A0_PASS" and all(audit.get("checks", {}).values()),
            "expert_not_accessed": access.get("expert_masks_read") is False and access.get("expert_directory_traversed") is False,
            "development_not_opened": access.get("development_model_outputs_read") is False,
        })
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "expert_data_accessed": False}
    write_json(A0_ROOT / "validation_receipt.json", result)
    return result


def validate_a1() -> dict[str, Any]:
    required = [
        A1_ROOT / "protocol.json", A1_ROOT / "protocol_hash.txt", A1_ROOT / "split_manifest.json",
        A1_ROOT / "label_subset_manifest.json", A1_ROOT / "threshold_freeze.json", A1_ROOT / "development_open_receipt.json",
        A1_ROOT / "per_section.csv", A1_ROOT / "per_annotator.csv", A1_ROOT / "per_stratum.csv",
        A1_ROOT / "operator_diagnostics.json", A1_ROOT / "metrics.json", A1_ROOT / "bootstrap.json",
        A1_ROOT / "ANZA_LEADS_A1_REPORT.md",
    ]
    checks = {f"exists:{path.name}": path.is_file() for path in required}
    if not all(checks.values()):
        result = {"status": "FAIL", "checks": checks}
        write_json(A1_ROOT / "validator.json", result)
        return result
    metrics = json.loads((A1_ROOT / "metrics.json").read_text())
    freeze = json.loads((A1_ROOT / "threshold_freeze.json").read_text())
    freeze_sha = freeze.pop("freeze_sha256")
    with (A1_ROOT / "per_section.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in LEADS_VARIANTS}
    summaries = {
        variant: {key: float(np.mean([float(row[key]) for row in local])) for key in (
            "dice", "cldice", "fragmentation", "unknown_white_foreground_fraction"
        )} for variant, local in by_variant.items()
    }
    l2, l3 = summaries["L2_generic_aniso"], summaries["L3_anza_hs"]
    dice_delta = l3["dice"] - l2["dice"]
    cldice_delta = l3["cldice"] - l2["cldice"]
    frag_ratio = l3["fragmentation"] / l2["fragmentation"] if l2["fragmentation"] > 0 else (0.0 if l3["fragmentation"] == 0 else math.inf)
    white_ratio = l3["unknown_white_foreground_fraction"] / l2["unknown_white_foreground_fraction"] if l2["unknown_white_foreground_fraction"] > 0 else (0.0 if l3["unknown_white_foreground_fraction"] == 0 else math.inf)
    gate = PROTOCOL["gate"]
    recomputed_pass = (
        dice_delta >= float(gate["dice_delta_min"])
        and white_ratio <= float(gate["unknown_white_foreground_ratio_max"])
        and (cldice_delta >= float(gate["cldice_delta_min"]) or frag_ratio <= float(gate["fragmentation_ratio_max"]))
    )
    valid_statuses = {
        "ANZA_LOW_LABEL_MECHANISM_PASS", "ANZA_LOW_LABEL_SCALE_REPAIR_AUTHORIZED", "STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL"
    }
    checks.update({
        "protocol_hash": (A1_ROOT / "protocol_hash.txt").read_text().strip() == protocol_hash(),
        "threshold_freeze_valid": freeze_sha == canonical_hash(freeze) and freeze.get("development_data_accessed") is False,
        "all_variants_present": all(len(by_variant[variant]) > 0 for variant in LEADS_VARIANTS),
        "equal_development_sections": len({tuple(int(row["section_id"]) for row in by_variant[variant]) for variant in LEADS_VARIANTS}) == 1,
        "gate_recomputed": bool(metrics["gate_pass"]) == bool(recomputed_pass),
        "valid_research_status": metrics["status"] in valid_statuses,
        "expert_not_accessed": metrics.get("expert_data_accessed") is False,
        "downstream_locks": all(metrics.get(key) is False for key in (
            "seeds_42_43_opened", "anza_ms_implemented", "ssl_opened", "domain_shift_opened", "oof_opened", "lira_opened"
        )),
    })
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "research_status": metrics["status"],
        "checks": checks, "recomputed": {"dice_delta": dice_delta, "cldice_delta": cldice_delta,
        "fragmentation_ratio": frag_ratio, "unknown_white_ratio": white_ratio, "gate_pass": recomputed_pass},
        "expert_data_accessed": False,
    }
    write_json(A1_ROOT / "validator.json", result)
    return result
