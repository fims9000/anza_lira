"""Freeze the best predeclared V2 candidate after validation-only development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from synthetic.evaluation import evaluation_protocol_hash
from synthetic.experiment_matrix import development_matrix, protocol_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_validation_candidate(study_root: Path) -> dict[str, Any]:
    validation_root = study_root / "synthetic" / "validation"
    development_root = study_root / "synthetic" / "development"
    by_id = {spec.candidate_id: spec for spec in development_matrix()}
    results = {
        candidate_id: json.loads(
            next(validation_root.glob(f"{candidate_id}-{spec.run_hash}.json")).read_text()
        )
        for candidate_id, spec in by_id.items()
    }
    baseline = results["C0"]["metrics"]
    decisions = {}
    for candidate_id in ("C1", "C2", "C3", "C4", "C5"):
        metrics = results[candidate_id]["metrics"]
        pairing_gain = metrics["branch_pairing_accuracy"] - baseline["branch_pairing_accuracy"]
        merge_reduction = (
            (baseline["false_merge_rate"] - metrics["false_merge_rate"])
            / baseline["false_merge_rate"]
            if baseline["false_merge_rate"]
            else 0.0
        )
        switch_reduction = (
            (baseline["identity_switch_rate"] - metrics["identity_switch_rate"])
            / baseline["identity_switch_rate"]
            if baseline["identity_switch_rate"]
            else 0.0
        )
        structural_numeric = pairing_gain >= 0.05 or merge_reduction >= 0.15 or switch_reduction >= 0.15
        visible_safe = metrics["visible_cldice"] >= baseline["visible_cldice"] - 0.01
        latent_safe = metrics["latent_cldice"] >= baseline["latent_cldice"] - 0.01
        continuation_coverage_safe = metrics["continuation_recall"] >= 0.90 * baseline["continuation_recall"]
        completion_control_safe = True
        if candidate_id in {"C4", "C5"}:
            completion_control_safe = (
                metrics["gap_recovery_rate"] >= baseline["gap_recovery_rate"]
                and metrics["false_bridge_rate"] < baseline["false_bridge_rate"]
            )
        decisions[candidate_id] = {
            "pairing_gain_absolute": pairing_gain,
            "false_merge_relative_reduction": merge_reduction,
            "identity_switch_relative_reduction": switch_reduction,
            "original_numeric_structural_condition": structural_numeric,
            "visible_cldice_safe": visible_safe,
            "latent_cldice_safe": latent_safe,
            "continuation_coverage_safe": continuation_coverage_safe,
            "positive_negative_gap_control_safe": completion_control_safe,
            "overall_gate": bool(
                structural_numeric
                and visible_safe
                and latent_safe
                and continuation_coverage_safe
                and completion_control_safe
            ),
        }

    frozen_id = "C3"
    frozen_spec = by_id[frozen_id]
    checkpoint = development_root / f"{frozen_id}-{frozen_spec.run_hash}" / "checkpoint-last.pt"
    freeze_payload = {
        "status": "FROZEN_NEGATIVE_DEVELOPMENT_RESULT",
        "quality_gate": "NOT_MET",
        "reason": (
            "No V2 candidate preserved continuation coverage while meeting a structural improvement; "
            "gap candidates also failed negative-gap control and latent clDice safety."
        ),
        "frozen_candidate_id": frozen_id,
        "frozen_model": frozen_spec.model,
        "frozen_objectives": list(frozen_spec.objectives),
        "selection_rule": (
            "Among C1-C3 without catastrophic latent completion degradation, select maximum validation "
            "visible Dice; C3 also has the highest branch continuation F1 of that subset."
        ),
        "development_protocol_hash": protocol_hash(),
        "evaluation_protocol_hash": evaluation_protocol_hash(),
        "run_hash": frozen_spec.run_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_metrics": results[frozen_id]["metrics"],
        "candidate_gate_decisions": decisions,
        "test_stream_before_freeze": "FROZEN_UNOPENED",
        "test_open_authorization": "ONE_EVALUATION_AFTER_THIS_FREEZE",
    }
    encoded = json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode()
    freeze_payload["freeze_sha256"] = hashlib.sha256(encoded).hexdigest()
    output = study_root / "synthetic" / "frozen_v2.json"
    output.write_text(json.dumps(freeze_payload, indent=2, sort_keys=True) + "\n")
    return freeze_payload
