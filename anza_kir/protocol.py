"""Frozen ANZA-KIR IR0--IR2 protocol."""

from __future__ import annotations

import hashlib
import json

from .benchmark import BASE_PRETRAIN_SIZE, NATURAL_SIZES, POOL_SIZES, SEEDS, VERSION
from .model import KIR_VARIANTS


def protocol() -> dict[str, object]:
    return {
        "version": VERSION,
        "parent_status": "STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED",
        "seed": 41,
        "streams": {"base_pretrain": BASE_PRETRAIN_SIZE, "natural": NATURAL_SIZES, "candidate_pools": POOL_SIZES, "seeds": SEEDS},
        "hard_mining": {"selector": "lowest margin 20% within each of five mechanism types", "anza_features_used": False, "minimum_total_pool": 50_000, "dev_hard_minimum": 2000, "valid_pair_error": [0.10, 0.40]},
        "matrix": list(KIR_VARIANTS),
        "base_training": {"epochs": 15, "batch_size": 16, "learning_rate": 0.001, "weight_decay": 0.0001},
        "residual_training": {"epochs": 15, "batch_size": 8, "learning_rate": 0.001, "weight_decay": 0.0001, "natural_samples": 2048, "hard_samples": 2048, "correction_l2": 0.0001},
        "feature": {"input": "directly supervised sigmoid evidence probability", "patch": 17, "orientation_count": 8, "static_width": 104, "dynamic_width": 104, "readout": [208, 32, 16], "normalization": "IR1 frozen evidence on residual-train only"},
        "loss": "segmentation + 0.10 orientation + 0.10 evidence during IR1; frozen IR2 segmentation + 1e-4 correction L2",
        "bootstrap": {"resamples": 10_000, "unit": "independent scene", "seed": 3_191_000_041},
        "gates": {
            "practical_pair_error": "R3 <= 0.70 R0",
            "pixel_safety": "Dice_R3 >= Dice_R0 - 0.005",
            "natural_topology": "clDice_R3-R0 >= 0.010 OR fragmentation_R3 <= 0.90 R0",
            "kolmogorov": "PairError_R3 <= 0.85 R2 AND paired margin CI lower > 0",
            "anosov": "PairError_R3 <= 0.90 R1 AND paired margin CI lower > 0",
        },
        "locks": {"confirm": True, "cracks": True, "expert": True, "seeds_42_43": True, "controlled_unfreezing": True},
    }


def protocol_hash(value: dict[str, object] | None = None) -> str:
    return hashlib.sha256(json.dumps(value or protocol(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
