"""Protocol constants frozen before SS1 robustness metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/anza_lira_ss_v1"
PROTOCOL_ID = "ANZA_LIRA_CRACKS_STRUCTURAL_STABILITY_V1"
FAMILIES = ("gain", "noise", "bandlimit", "phase", "warp")
SEVERITIES = (1, 2, 3)
TRAIN_SEVERITIES = (1, 2)

PROTOCOL = {
    "protocol_id": PROTOCOL_ID,
    "authorized_phases": ["SS0_AUDIT_FREEZE", "SS1_PERTURBATION_VALIDATION"],
    "dataset": "existing local CRACKS only; no downloads",
    "split_by_sorted_valid_rank": {
        "SS_TRAIN": [1, 220],
        "BUFFER_1": [221, 230],
        "SS_CALIBRATION": [231, 270],
        "BUFFER_2": [271, 280],
        "SS_DEVELOPMENT": [281, 330],
        "BUFFER_3": [331, 340],
        "SS_CONFIRM": [341, None],
    },
    "partial_label_semantics": {
        "blue": [1.0, 1.0], "green": [1.0, 0.5],
        "orange": [0.0, 1.0], "white": [0.0, 0.0],
    },
    "agreement": "abs(2*W_pos/(W_pos+W_neg)-1)^2 * min(1,(W_pos+W_neg)/3); zero when unlabeled",
    "families": list(FAMILIES),
    "severities": list(SEVERITIES),
    "training_severities": list(TRAIN_SEVERITIES),
    "severity_values": {
        "gain": {"1": [0.90, 1.10], "2": [0.80, 1.20], "3": [0.65, 1.35]},
        "noise": {"1": 30.0, "2": 20.0, "3": 12.0},
        "bandlimit": {"1": 0.5, "2": 1.0, "3": 1.5},
        "phase": {"1": 5.0, "2": 10.0, "3": 20.0},
        "warp": {"1": 0.5, "2": 1.0, "3": 2.0},
    },
    "warp_validity": {"det_min": 0.75, "det_max": 1.25, "condition_max": 1.5, "maximum_attempts": 32},
    "seed_formula": "SHA256(protocol_id,section_id,crop_id,family,severity,view_index)",
    "perturbation_point": "after historical normalization; never per-image renormalize afterward",
    "ss1_model": "frozen historical T1 U-Net seed41 only",
    "ss1_threshold": "historical frozen T1 seed41 threshold",
    "ss1_gate": "finite + deterministic + valid warp Jacobian + exact transformed palette + panels; no degradation threshold",
    "locks": {"B0": True, "B1": True, "B2": True, "B3": True, "seeds_42_43": True, "LIRA": True, "confirm": True},
    "claim_boundary": "controlled label-preserving perturbations on CRACKS; no cross-survey, natural-gap, expert-instance, or Anosov-dynamics claim",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)

