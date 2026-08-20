"""Frozen zero-training E0/E1 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .e1_bench import benchmark_config
from .kernels import METHODS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "anza_ek" / "e0_e1"
FREEZE_ROOT = RESULT_ROOT / "freeze"
VERSION = "ANZA_EK_E0_E1_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "question": "Do finite Cat-map Koopman correlations provide an identifiable elongated-structure signal beyond static anisotropy and a measure-preserving nonhyperbolic shear?",
        "cat_matrix": [[2, 1], [1, 1]],
        "shear_control_matrix": [[1, 1], [0, 1]],
        "grid_size_e0": 257,
        "exact_discrete_grid_size": 128,
        "kernel_size": 65,
        "K": 3,
        "seed_sigma": 0.085,
        "orientation_count": 8,
        "methods": list(METHODS),
        "score": "max_nonzero_abs_c_minus_abs_c0 + 0.25*mean_abs(c_k-c_-k) + 0.10*var(c)",
        "e1_benchmark": benchmark_config(),
        "primary_metrics": ["matched_ranking", "auroc", "tpr_at_fpr05", "fisher_separation", "perturbation_score_correlation"],
        "gate": {
            "task_gain_tpr_or_ranking": 0.08,
            "minimum_passing_tasks": 2,
            "macro_clean_ranking_noninferiority": -0.02,
            "macro_perturbed_ranking_noninferiority": -0.02,
            "macro_perturbation_correlation_noninferiority": -0.02,
            "strongest_control": "highest macro clean matched ranking among E1-0/E1-1/E1-2",
        },
        "training_performed": False,
        "learned_classifier": False,
        "E2_opened": False,
        "conjugacy_opened": False,
        "confirm_created": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }


def freeze_protocol(output_root: Path = FREEZE_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = output_root / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("ANZA-EK E0/E1 protocol drift")
    path.write_text(encoded)
    (output_root / "protocol_hash.txt").write_text(canonical_hash(payload) + "\n")
    return payload
