"""Frozen ANZA-KS K0/K1 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmark.matched_generator import benchmark_manifest
from .constants import FEATURE_WIDTH, ORIENTATION_COUNT
from .features import METHODS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "anza_ks" / "k0_k1"
FREEZE_ROOT = RESULT_ROOT / "freeze"
VERSION = "ANZA_KS_K0_K1_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "question": "Do finite symbolic itineraries under exact Cat dynamics add higher-order information after frozen static anisotropic information is matched?",
        "cat_matrix": [[2, 1], [1, 1]],
        "shear_matrix": [[1, 1], [0, 1]],
        "grid_size": 17,
        "partition": "four fixed equal torus quadrants; finite partition, not claimed Markov",
        "K": 4,
        "word_lengths": [1, 2, 3, 4],
        "predictive_length": 2,
        "orientation_count": ORIENTATION_COUNT,
        "feature_width": FEATURE_WIDTH,
        "methods": list(METHODS),
        "benchmark": benchmark_manifest(),
        "readout": {
            "type": "task-specific identical StandardScaler + L2 logistic regression",
            "C": 1.0,
            "solver": "liblinear",
            "random_state": 17,
            "max_iter": 500,
            "fit_pairs_per_task": 1536,
            "calibration_pairs_per_task": 512,
            "development_pairs_per_task": 1024,
            "threshold_policy": "maximize TPR subject to calibration FPR <=0.05; development curve metric remains threshold-free",
            "hyperparameter_sweep": False,
        },
        "bootstrap": {"resamples": 10_000, "seed": 941_019, "unit": "paired synthetic example within task"},
        "gate": {
            "static_auroc_min": 0.45,
            "static_auroc_max": 0.60,
            "full_task_tpr_or_ranking_gain": 0.08,
            "minimum_full_gain_tasks": 3,
            "kolmogorov_macro_gain": 0.04,
            "kolmogorov_bootstrap_lower_gt_zero": True,
            "cat_vs_shear_macro_noninferiority": -0.02,
            "cat_vs_shear_minimum_winning_tasks": 2,
        },
        "training_scope": "fixed tiny logistic readouts only",
        "segmentation_training_performed": False,
        "K2_opened": False,
        "confirm_generated_hash_only": True,
        "confirm_evaluated": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }


def freeze_protocol() -> dict[str, Any]:
    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = FREEZE_ROOT / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("ANZA-KS protocol drift")
    path.write_text(encoded)
    (RESULT_ROOT / "protocol.json").write_text(encoded)
    digest = canonical_hash(payload)
    (FREEZE_ROOT / "protocol_hash.txt").write_text(digest + "\n")
    (RESULT_ROOT / "protocol_hash.txt").write_text(digest + "\n")
    return payload
