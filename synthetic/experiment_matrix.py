"""Frozen candidate and fairness contracts for controlled development."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


COMMON_PROTOCOL = {
    "split": "crossing_trace_bench_validation",
    "selection_seed": 42,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 4,
    "microbatch_size": 1,
    "gradient_accumulation": 4,
    "epoch_budget": 20,
    "image_size": 128,
    "development_train_indices": "train[0:256] fixed",
    "development_validation_indices": "validation[0:256] fixed",
    "visible_segmentation_loss": "bce_plus_dice",
    "threshold_procedure": "validation_visible_dice",
    "evaluation_code": "synthetic.structural_metrics.compute_structural_metrics",
    "test_stream": "FROZEN_UNOPENED",
}


@dataclass(frozen=True)
class SyntheticRunSpec:
    candidate_id: str
    model: str
    objectives: tuple[str, ...]
    comparison_family: str
    seed: int = 42
    kappa_theta: float = 4.0
    kappa_direction: float = 4.0

    @property
    def run_hash(self) -> str:
        payload = {"run": asdict(self), "common_protocol": COMMON_PROTOCOL}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def development_matrix() -> tuple[SyntheticRunSpec, ...]:
    visible = ("visible_bce_dice",)
    return (
        SyntheticRunSpec("B0", "unet", visible, "architecture_only"),
        SyntheticRunSpec("B1", "deformable_unet", visible, "architecture_only"),
        SyntheticRunSpec("C0", "anza_v1", visible, "architecture_only"),
        SyntheticRunSpec("C1", "anza_v2a", visible, "architecture_only"),
        SyntheticRunSpec("C2", "anza_v2b", visible, "architecture_only"),
        SyntheticRunSpec("C3", "anza_v2b", visible + ("route",), "structural_supervision"),
        SyntheticRunSpec(
            "C4",
            "anza_v2_full",
            visible + ("route", "positive_negative_gap"),
            "structural_supervision",
        ),
        SyntheticRunSpec(
            "C5",
            "anza_v2_full",
            visible + ("route", "positive_negative_gap", "cone"),
            "structural_supervision",
        ),
    )


def protocol_hash() -> str:
    encoded = json.dumps(COMMON_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
