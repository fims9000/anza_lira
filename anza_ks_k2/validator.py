"""Final fail-closed validator for the bounded K1.5/K2 seed-41 cycle."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import FREEZE, RESULT, source_manifest


def validate(*, require_results: bool = True) -> dict[str, object]:
    receipt = json.loads((FREEZE / "pretraining_receipt.json").read_text())
    if receipt["source_sha256"] != source_manifest()["sha256"]: raise AssertionError("K2 source drift")
    if receipt["confirm_evaluated"]: raise AssertionError("confirm was opened")
    result: dict[str, object] = {"pretraining_status": receipt["status"], "source_sha256": receipt["source_sha256"], "confirm_opened": False}
    if require_results:
        metrics = json.loads((RESULT / "metrics.json").read_text())
        if set(metrics["variants"]) != {"M0_backbone", "M1_static", "M2_shear_ks", "M3_cat_raw", "M4_anza_ks"}: raise AssertionError("K2 matrix incomplete")
        for value in metrics["variants"].values():
            if value["run"]["epoch"] != 15 or value["run"]["seed"] != 41: raise AssertionError("K2 training budget mismatch")
        if metrics["confirm_opened"] or metrics["seeds_42_43_opened"] or metrics["cracks_accessed"] or metrics["expert_accessed"]: raise AssertionError("downstream lock violated")
        result["research_status"] = metrics["status"]
    return result
