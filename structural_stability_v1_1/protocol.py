"""Frozen V1.1 amendment constants; no model result may alter these values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from structural_stability_v1.protocol import PROTOCOL_ID as PARENT_PROTOCOL_ID, ROOT


PROTOCOL_ID = "ANZA_LIRA_CRACKS_STRUCTURAL_STABILITY_V1_1"
RESULT_ROOT = ROOT / "results/anza_lira_ss_v1_1"
PARENT_RESULT_ROOT = ROOT / "results/anza_lira_ss_v1"
AMENDMENT_SOURCE = Path(
    "/home/lebedeffson/.codex/attachments/ba84a96c-82b9-4beb-81c5-573bfd1173df/pasted-text.txt"
)
VARIANTS = ("B0", "B1", "B2", "B3")
SEEDS = (41, 42, 43)

PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "parent_protocol_id": PARENT_PROTOCOL_ID,
    "authorized_phase": "SS1_5_PRETRAINING_HARDENING",
    "parent_split_sha256": "43a3fb7716d5ff9e56c7da9a78f2127c20f8d13ba27d7e5576ac493176045671",
    "parent_artifacts_immutable": True,
    "variants": list(VARIANTS),
    "seeds": list(SEEDS),
    "planned_training_jobs": 12,
    "training": {
        "initialization": "fresh standard initialization; historical H0 forbidden",
        "normalization": "SS_TRAIN image pixels only",
        "epochs": 36,
        "historical_optimizer_updates": 1980,
        "planned_optimizer_updates": 1980,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "crop_size": 256,
        "foreground_crop_probability": 0.7,
        "effective_batch_size": 4,
        "annotators_per_section": 4,
        "perturbation_families": ["gain", "noise", "bandlimit", "phase", "warp"],
        "training_severities": [1, 2],
        "evaluation_only_severity": 3,
    },
    "geometry_target": {
        "positive_field": "crowd_agreement * crowd_probability",
        "positive_smoothing_sigma": 1.5,
        "structure_tensor_sigma": 2.0,
        "minimum_crowd_probability": 0.75,
        "minimum_agreement": 0.25,
        "minimum_coherence": 0.20,
        "strength": "0.35 * coherence",
        "annotators": "TRAIN_ANNOTATORS nonexpert only",
    },
    "metric": {
        "representation": "R(theta) diag(exp(2(m+d)),exp(2(m-d))) R(theta)^T",
        "B2": {"d": "0.5*sigmoid", "m": "0.5*tanh", "determinant": "free exp(4m)"},
        "B3": {"d": "0.5*sigmoid", "m": 0.0, "determinant": 1.0},
        "transport": "C'=Abar C Abar^T; Abar=A/sqrt(det A); A=(Dphi_output_to_input)^-1",
        "log_eigenvalue_clamp": [1e-4, 1e4],
        "sidecar_locations": ["decoder_1_4", "decoder_1_2"],
        "sidecar_hidden_width": 16,
    },
    "loss_weights": {"probability": 0.20, "topology": 0.20, "axis": 0.05, "strength": 0.05, "equivariance": 0.05},
    "locks": {
        "training": True, "development": True, "confirm": True, "expert_pixels": True,
        "H0_initialization": True, "LIRA": True,
    },
    "claim_boundary": "pre-training implementation validity only; no robustness or Anosov advantage result",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)
