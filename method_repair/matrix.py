"""Predeclared bounded A0-A4 synthetic development matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from synthetic.crossing_trace_bench_v2 import benchmark_v2_config


COMMON_PROTOCOL = {
    "split": "crossing_trace_bench_v2_validation",
    "seed": 42,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "gradient_accumulation": 4,
    "epoch_budget": 20,
    "image_size": 128,
    "train_indices": "train[0:256]",
    "validation_indices": "validation[0:256]",
    "visible_loss": "bce_plus_dice",
    "mode_set_weight": 0.2,
    "mode_route_weight": 0.2,
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "old_test_stream": "IMMUTABLE_NOT_USED",
    "new_test_stream": "LOCKED_UNOPENED_UNTIL_CANDIDATE_FREEZE",
    "expert_access": "FORBIDDEN",
    "benchmark_v2_sha256": benchmark_v2_config()["sha256"],
    "mechanism_gate": {
        "neff_junction_minus_straight_bootstrap_ci_low_gt": 0.0,
        "ambiguity_junction_minus_straight_bootstrap_ci_low_gt": 0.0,
        "route_average_precision_minimum_gain_over_old_c3": 0.01,
        "old_c3_route_average_precision": 0.7327459922609177,
        "route_excess_over_chance_minimum": 0.05,
        "route_entropy_normalized_maximum": 0.95,
        "false_bridge_rate_maximum_exclusive": 1.0,
        "visible_dice_noninferiority_margin": 0.01,
        "visible_cldice_noninferiority_margin": 0.01,
    },
    "selection_composite": "route_average_precision + route_mrr - route_entropy_normalized + neff_delta + ambiguity_delta - false_bridge_rate; only all-gate candidates eligible",
}


@dataclass(frozen=True)
class MethodRepairSpec:
    candidate_id: str
    model: str
    use_ambiguity_gate: bool
    direct_mode_supervision: bool
    routing_kernel_size: int
    seed: int = 42

    @property
    def run_hash(self) -> str:
        payload = {"spec": asdict(self), "protocol": COMMON_PROTOCOL}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def synthetic_matrix() -> tuple[MethodRepairSpec, ...]:
    return (
        MethodRepairSpec("A0", "anza_v1", False, False, 3),
        MethodRepairSpec("A1", "repaired", False, False, 3),
        MethodRepairSpec("A2", "repaired", True, False, 3),
        MethodRepairSpec("A3", "repaired", True, True, 3),
        MethodRepairSpec("A4", "repaired", True, True, 5),
    )


def protocol_hash() -> str:
    encoded = json.dumps(COMMON_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
