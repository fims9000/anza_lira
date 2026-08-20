"""Frozen K2 seed-41 protocol."""

from __future__ import annotations

import hashlib
import json

from .benchmark import SPLIT_SEEDS, SPLIT_SIZES, VERSION
from .model import VARIANTS


def protocol() -> dict[str, object]:
    return {
        "version": "ANZA_KS_K2_V1",
        "parent_benchmark": VERSION,
        "splits": SPLIT_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "matrix": list(VARIANTS),
        "seed": 41,
        "epochs": 15,
        "batch_size": 8,
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
        "loss": "BCE+softDice+0.25softclDice; structured add 0.10 orientation +0.05 balanced occupancy",
        "orientation": {"count": 8, "width": 0.20, "visible_only": True},
        "feature": {"patch": 17, "width": 104, "decoder_scale": "1/4", "normalization": "K2 train indices 0..255 only, frozen before training"},
        "calibration": "train-only stream; mechanism TargetRecall>=0.95",
        "bootstrap": {"resamples": 10000, "unit": "independent scene", "seed": 2_019_451},
        "gates": {
            "pixel_safety": "Dice_M4 >= Dice_M1 - 0.005",
            "mechanism": "FPR_M4@Recall95 <= 0.70 FPR_M1",
            "natural": "matched-precision clDice delta >= 0.010 OR fragmentation_M4 <= 0.90 fragmentation_M1",
            "kolmogorov": "FPR_M4 <=0.80 FPR_M3 OR TPR@FPR05 delta >=0.08; natural Dice noninferior",
            "anosov": "FPR_M4 < FPR_M2 with paired CI lower improvement >0; target >=10% relative",
        },
        "locks": {"confirm": True, "seeds_42_43": True, "cracks": True, "expert": True},
    }


def protocol_hash(value: dict[str, object] | None = None) -> str:
    encoded = json.dumps(value or protocol(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
