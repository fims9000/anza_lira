"""Pre-gradient frozen H0/H1 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .stress_bench import stressbench_config


ROOT = Path(__file__).resolve().parents[1]
A2_ROOT = ROOT / "results" / "anza_s" / "a2"
H0_ROOT = ROOT / "results" / "anza_hs" / "h0"
VERSION = "ANZA_HS_H0_H1_FROZEN_V1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    validator = json.loads((A2_ROOT / "validator.json").read_text())
    if validator.get("research_status") != "ANOSOV_COCYCLE_REDUNDANT_AT_ORACLE":
        raise ValueError("frozen Phase A2 status changed")
    return {
        "version": VERSION,
        "parent_a2_protocol_sha256": digest(A2_ROOT / "protocol.json"),
        "parent_a2_validator_sha256": digest(A2_ROOT / "validator.json"),
        "parent_a2_status": validator["research_status"],
        "question": "Does reciprocal local hyperbolicity outperform a capacity-matched free anisotropic orientation-bank operator?",
        "stressbench": stressbench_config(),
        "matrix": ["B0_backbone", "B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic"],
        "architecture": {"M": 8, "support": 9, "base_scale": 1.5, "lambda": 0.35, "placements": ["decoder_1_4", "decoder_1_2"], "gamma_init": 0.0},
        "orientation_target": {"sigma_theta": 0.20, "background_weight": 0.25, "source": "visible generator branch axes only"},
        "training": {"seed": 41, "epochs": 20, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 16, "train_samples": 352, "segmentation_loss": "BCE_plus_soft_Dice", "orientation_loss_weight": 0.10},
        "threshold": {"grid": [round(value, 2) for value in [0.20 + 0.05 * index for index in range(13)]], "calibration": "dev[0:44]", "gate": "dev[44:264]", "B3_rule": "closest calibration precision to B2 at its max-Dice threshold; tie highest Dice", "other_rule": "max calibration Dice; tie highest precision"},
        "gate": {"dice_noninferiority": -0.005, "cldice_gain": 0.015, "fragmentation_relative_max": 0.90, "primary_comparison": "B3_anza_hyperbolic minus B2_generic_aniso"},
        "allowed_development_alternative": {"base_scale": 2.0, "used": False},
        "training_allowed_after_h0_pass": True,
        "confirm_opened": False, "test_created": False, "cracks_accessed": False, "continuation_trained": False,
        "expert_accessed": False, "lambda_tuned": False, "M_tuned": False, "H2_opened": False,
    }


def freeze_protocol(output_root: Path = H0_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload(); encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = output_root / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("ANZA-HS H0/H1 protocol drift")
    path.write_text(encoded)
    (output_root / "protocol_hash.txt").write_text(canonical_hash(payload) + "\n")
    return payload
