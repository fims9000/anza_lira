"""Frozen C0--C3 candidate matrix for the independent affinity stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from synthetic.crossing_trace_bench_v4 import benchmark_v4_config


AFFINITY_PROTOCOL = {
    "version": "anza_structural_affinity_c0_c3_v1",
    "seed": 42,
    "optimizer": "adam",
    "stage1_epochs": 8,
    "stage2_epochs": 25,
    "base_learning_rate": 1e-4,
    "affinity_learning_rate": 1e-3,
    "gradient_accumulation": 4,
    "image_size": 128,
    "train_indices": "train_v4[0:512]",
    "validation_indices": "validation_v4[0:512]",
    "confirm_indices": "confirm_v4[0:512]",
    "visible_loss": "bce_plus_dice_plus_0.2_soft_cldice",
    "affinity_loss_weight": 0.2,
    "ranking_loss_weight": 0.2,
    "ranking_margin": 0.5,
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "benchmark_v4_sha256": benchmark_v4_config()["sha256"],
    "test_v4": "LOCKED_UNOPENED",
    "legacy_test": "IMMUTABLE_NOT_REUSED",
    "cracks": "FORBIDDEN_UNTIL_CONFIRM_PASS",
    "expert_access": "FORBIDDEN",
    "selection": "fixed C0-C3 gates; no C4",
}


@dataclass(frozen=True)
class AffinityRepairSpec:
    candidate_id: str
    independent_fuzzy: bool
    affinity: bool
    radius2: bool
    hard_ranking: bool
    seed: int = 42

    @property
    def run_hash(self) -> str:
        payload = {"spec": asdict(self), "protocol": AFFINITY_PROTOCOL}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def affinity_matrix() -> tuple[AffinityRepairSpec, ...]:
    return (
        AffinityRepairSpec("C0", False, False, False, False),
        AffinityRepairSpec("C1", True, False, False, False),
        AffinityRepairSpec("C2", True, True, False, False),
        AffinityRepairSpec("C3", True, True, True, True),
    )


def affinity_protocol_hash() -> str:
    return hashlib.sha256(json.dumps(AFFINITY_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def freeze_affinity_protocol(path: Path) -> dict:
    payload = {**AFFINITY_PROTOCOL, "protocol_hash": affinity_protocol_hash()}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError("affinity-repair protocol drift")
    path.write_text(encoded)
    return payload
