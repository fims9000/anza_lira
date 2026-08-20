"""Predeclared B0-B3 matrix for the bounded context-repair cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from synthetic.crossing_trace_bench_v3 import benchmark_v3_config


CONTEXT_PROTOCOL = {
    "version": "anza_context_repair_b0_b3_v1",
    "seed": 42,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "gradient_accumulation": 4,
    "epoch_budget": 25,
    "image_size": 128,
    "train_indices": "train_v3[0:512]",
    "validation_indices": "validation_v3[0:512]",
    "confirm_indices": "confirm_v3[0:512]",
    "visible_loss": "bce_plus_dice",
    "mode_set_weight": 0.2,
    "mode_route_weight": 0.2,
    "gate_weight": 0.2,
    "contrastive_route_weight": 0.2,
    "gap_weight": 0.2,
    "negative_gap_beta": 1.0,
    "route_temperature_candidates": [0.1, 0.2],
    "selected_route_temperature": 0.1,
    "gate_sigma_px": 2.5,
    "transport_kernel_size": 3,
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "benchmark_v3_sha256": benchmark_v3_config()["sha256"],
    "test_v3": "LOCKED_UNOPENED",
    "legacy_test": "IMMUTABLE_NOT_REUSED",
    "cracks": "FORBIDDEN_UNTIL_CONFIRM_PASS",
    "expert_access": "FORBIDDEN",
    "mechanism_gate": {
        "visible_noninferiority_margin": 0.005,
        "route_average_precision_minimum": 0.92,
        "route_entropy_normalized_maximum": 0.45,
        "orientation_error_median_deg_maximum": 12.0,
        "neff_ci95_low_minimum_exclusive": 0.0,
        "neff_mean_and_median_separation_minimum": 0.10,
        "membership_set_kl_maximum": 0.70,
        "gate_auroc_minimum": 0.85,
        "gate_delta_ci95_low_minimum_exclusive": 0.05,
        "negative_gap_count_minimum": 128,
        "false_bridge_rate_maximum": 0.50,
        "gap_recovery_rate_minimum": 0.85,
        "false_bridge_reduction_vs_b0_minimum": 0.20,
    },
    "selection": "all gates first; then route AP - entropy + N_eff delta + gate AUROC - false bridge",
}


@dataclass(frozen=True)
class ContextRepairSpec:
    candidate_id: str
    model: str
    contextual_gate: bool
    contrastive_route: bool
    paired_gap_loss: bool
    seed: int = 42

    @property
    def run_hash(self) -> str:
        encoded = json.dumps(
            {"spec": asdict(self), "protocol": CONTEXT_PROTOCOL},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def context_matrix() -> tuple[ContextRepairSpec, ...]:
    return (
        ContextRepairSpec("B0", "a3_pointwise", False, False, False),
        ContextRepairSpec("B1", "context", True, False, False),
        ContextRepairSpec("B2", "context", True, True, False),
        ContextRepairSpec("B3", "context", True, True, True),
    )


def context_protocol_hash() -> str:
    encoded = json.dumps(CONTEXT_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
