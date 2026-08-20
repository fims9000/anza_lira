"""Provenance and validation-only audit helpers for corrected evaluator v2.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import label

from synthetic.evaluation_corrected import (
    CORRECTED_EVALUATOR_VERSION,
    FALSE_BRIDGE_SENSITIVITY_THRESHOLDS,
    ORIGINAL_TEST_RANGE,
    PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD,
    REPLACEMENT_TEST_RANGE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corrected_evaluator_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).with_name("evaluation_corrected.py"),
        Path(__file__).with_name("structural_metrics_corrected.py"),
        Path(__file__),
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot_legacy_outputs(test_root: Path) -> dict[str, str]:
    """Hash existing legacy files so callers can prove they stayed immutable."""

    return {
        str(path.relative_to(test_root)): _sha256(path)
        for path in sorted(test_root.rglob("*"))
        if path.is_file()
    }


def assert_legacy_outputs_unchanged(test_root: Path, snapshot: Mapping[str, str]) -> None:
    if snapshot_legacy_outputs(test_root) != dict(snapshot):
        raise RuntimeError("Legacy synthetic test outputs changed during corrected evaluation")


def _disk(shape: tuple[int, int], point_xy: Sequence[float], radius: int = 3) -> np.ndarray:
    x, y = (int(round(float(value))) for value in point_xy)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - y) ** 2 + (xx - x) ** 2 <= radius**2


def validate_negative_gap_contract(target: Mapping[str, Any]) -> dict[str, Any]:
    records = [record for record in target["gaps"] if record["gap_type"] == "negative"]
    masks = np.asarray(target["negative_gap_masks"], dtype=bool)
    latent = np.asarray(target["latent_fault_mask"], dtype=bool)
    instances = np.asarray(target["instance_masks"], dtype=bool)
    instance_ids = [int(value) for value in target["fault_instance_ids"]]
    if len(records) != len(masks):
        raise ValueError("Negative gap records and masks differ in count")
    for record, mask in zip(records, masks):
        if not mask.any():
            raise ValueError("Negative gap candidate mask must be nonempty")
        if np.any(mask & latent):
            raise ValueError("Negative gap pixels must not enter latent_fault_mask")
        endpoint_memberships: list[set[int]] = []
        for endpoint in record["endpoint_xy"]:
            disk = _disk(latent.shape, endpoint)
            endpoint_memberships.append(
                {instance_id for instance_id, instance in zip(instance_ids, instances) if np.any(instance & disk)}
            )
        if any(not membership for membership in endpoint_memberships):
            raise ValueError("Each negative-gap endpoint must touch a known latent instance")
        if endpoint_memberships[0] & endpoint_memberships[1]:
            raise ValueError("Negative-gap endpoints must have no common latent instance")
    return {"status": "PASS", "negative_gap_count": len(records)}


def _endpoint_connected(binary: np.ndarray, endpoints: Sequence[Sequence[float]]) -> bool:
    components, _ = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    memberships = []
    for endpoint in endpoints:
        values = np.unique(components[_disk(binary.shape, endpoint)])
        memberships.append({int(value) for value in values if int(value)})
    return bool(memberships[0] & memberships[1])


def audit_gap_predictions(
    target: Mapping[str, Any],
    completion_probability: np.ndarray,
) -> dict[str, Any]:
    """Audit positive/negative gaps without selecting a threshold on test."""

    probability = np.asarray(completion_probability, dtype=np.float64)
    latent = np.asarray(target["latent_fault_mask"], dtype=bool)
    if probability.shape != latent.shape or not np.isfinite(probability).all():
        raise ValueError("Completion probability must be finite and match target HxW")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Completion probability must be in [0, 1]")
    validate_negative_gap_contract(target)
    output: dict[str, Any] = {
        "primary_coverage_threshold": PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD,
        "sensitivity_thresholds": list(FALSE_BRIDGE_SENSITIVITY_THRESHOLDS),
        "threshold_selection_permitted": False,
    }
    for gap_type, mask_key in (("positive", "positive_gap_masks"), ("negative", "negative_gap_masks")):
        records = [record for record in target["gaps"] if record["gap_type"] == gap_type]
        masks = np.asarray(target[mask_key], dtype=bool)
        rows = []
        for record, mask in zip(records, masks):
            row: dict[str, Any] = {
                "mean_probability": float(probability[mask].mean()),
                "max_probability": float(probability[mask].max()),
            }
            for threshold in FALSE_BRIDGE_SENSITIVITY_THRESHOLDS:
                binary = probability >= threshold
                suffix = f"{threshold:.2f}"
                row[f"coverage_at_{suffix}"] = float(binary[mask].mean())
                row[f"connected_at_{suffix}"] = _endpoint_connected(binary, record["endpoint_xy"])
            rows.append(row)
        output[gap_type] = {"count": len(rows), "rows": rows}
    primary = f"{PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD:.2f}"
    negatives = output["negative"]["rows"]
    output["false_bridge_rate_at_fixed_0_5"] = (
        float(np.mean([row[f"coverage_at_{primary}"] >= 0.5 and row[f"connected_at_{primary}"] for row in negatives]))
        if negatives else 0.0
    )
    return output


def false_bridge_verdict(audits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Declare saturation without turning sensitivity thresholds into tuning."""

    rates = {
        method: float(audit["false_bridge_rate_at_fixed_0_5"])
        for method, audit in audits.items()
    }
    saturated = bool(rates) and all(np.isclose(value, 1.0) for value in rates.values())
    return {
        "status": (
            "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE"
            if saturated
            else "FALSE_BRIDGE_ENDPOINT_RETAINS_DISCRIMINATIVE_RANGE"
        ),
        "fixed_threshold": PRIMARY_FALSE_BRIDGE_COVERAGE_THRESHOLD,
        "method_rates": rates,
        "sensitivity_used_for_threshold_selection": False,
        "eligible_for_positive_mechanism_claim": not saturated,
    }


def freeze_corrected_evaluator(
    study_root: Path,
    *,
    model_checkpoint_hashes: Mapping[str, str],
    visible_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Create the replacement-stream receipt before any replacement evaluation."""

    root = Path(study_root) / "synthetic" / "replacement_confirmation"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "freeze.json"
    payload = {
        "status": "FROZEN_BEFORE_OPENING",
        "reason": "evaluator defect detected after original test opening",
        "corrected_evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "corrected_evaluator_sha256": corrected_evaluator_hash(),
        "original_test_indices": list(ORIGINAL_TEST_RANGE),
        "original_test_status": "ALREADY_OPENED",
        "replacement_test_indices": list(REPLACEMENT_TEST_RANGE),
        "replacement_test_status": "PREVIOUSLY_UNOPENED",
        "model_checkpoint_hashes": dict(sorted(model_checkpoint_hashes.items())),
        "visible_thresholds": {key: float(value) for key, value in sorted(visible_thresholds.items())},
        "topology_assignment_rule": "X perfect matching; T one pair; Y shared hub; symmetric geometric-mean route score",
        "no_tuning_after_opening": True,
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError("Existing corrected evaluator freeze does not match current contract")
        return {**existing, "action": "SKIP"}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "action": "CREATE"}


def run_validation_audit(study_root: Path, *, audit: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """Persist caller-computed validation-only audit under the corrected path."""

    root = Path(study_root) / "synthetic" / "evaluator_audit"
    root.mkdir(parents=True, exist_ok=True)
    result = {"status": "COMPLETE", "split": "validation", "test_samples_opened": 0, **dict(audit or {})}
    (root / "validation_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_legacy_test_reanalysis(study_root: Path, *, reanalysis: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """Write post-hoc evidence outside the immutable legacy test directory."""

    study_root = Path(study_root)
    legacy_root = study_root / "synthetic" / "test"
    before = snapshot_legacy_outputs(legacy_root)
    root = study_root / "synthetic" / "evaluator_audit" / "legacy_test_reanalysis"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "POSTHOC_REANALYSIS_NOT_CONFIRMATORY",
        "indices": list(ORIGINAL_TEST_RANGE),
        "legacy_snapshot_sha256": before,
        **dict(reanalysis or {}),
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert_legacy_outputs_unchanged(legacy_root, before)
    return result


def run_replacement_confirmation(study_root: Path, *, confirmation: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    """Fail closed unless the corrected evaluator was frozen before opening."""

    root = Path(study_root) / "synthetic" / "replacement_confirmation"
    freeze_path = root / "freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("Corrected evaluator freeze must exist before replacement opening")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("corrected_evaluator_sha256") != corrected_evaluator_hash():
        raise RuntimeError("Corrected evaluator changed after freeze")
    summary_path = root / "summary.json"
    result = {
        "status": "REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT",
        "indices": list(REPLACEMENT_TEST_RANGE),
        "freeze_sha256": _sha256(freeze_path),
        "no_retraining": True,
        **dict(confirmation or {}),
    }
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing != result:
            raise RuntimeError("Replacement confirmation is immutable once written")
        return {**existing, "action": "SKIP"}
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "action": "CREATE"}
