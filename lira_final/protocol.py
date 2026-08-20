"""Frozen protocol constants for ANZA-LIRA Final Seismic Endgame V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/lira_final"
HELDOUT_ANNOTATORS = ("novice12", "novice20", "practitioner2")
TRAIN_ANNOTATORS = tuple(
    f"novice{index:02d}" for index in range(1, 27) if index not in (12, 20)
) + tuple(f"practitioner{index}" for index in range(1, 9) if index != 2)

# These ranges were frozen before natural-gap counts or model scores were read.
SPLIT_RANGES = {
    "relation_train": (1, 200),
    "dense_calibration": (203, 240),
    "lira_calibration": (243, 290),
    "lira_development": (293, 344),
    "lira_confirm": (347, 400),
}

PROTOCOL = {
    "version": "ANZA_LIRA_FINAL_SEISMIC_ENDGAME_V1",
    "phases_authorized": ["F0_FINAL_FREEZE", "F1_REAL_GAP_AUDIT", "F2_REAL_CANDIDATE", "F3_REAL_RELATION_S41"],
    "dense": {
        "source": "mean frozen T1 U-Net seeds 41/42/43",
        "threshold_selection": "maximize mean heldout-annotator clDice subject to micro precision >=0.75",
        "threshold_candidates": [round(value / 100, 2) for value in range(5, 96, 5)],
        "soft_ratios": [0.4, 0.6, 0.8],
        "immutable": True,
    },
    "splits": {name: list(bounds) for name, bounds in SPLIT_RANGES.items()},
    "split_buffer_sections": 2,
    "trace_identity": "ordered non-junction skeleton segments from individual nonexpert raster annotations",
    "natural_gap": {"minimum": 6, "maximum": 48, "visible_context": 8},
    "candidate": {"k": 12, "distance": [6.0, 68.0], "angle_degrees": 78.0, "landing_band": 12},
    "p0": {
        "architecture": "path_completion.pair_classifier.EndpointPairClassifier",
        "initialization": "results/path_completion/pair_classifier/checkpoint.pt",
        "seed": 41,
        "epochs": 120,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "ranking_weight": 0.5,
        "ranking_temperature": 0.2,
        "batch_sources": 32,
        "artificial_gap_lengths": [8, 12, 16, 24, 32, 40],
        "maximum_negatives": 3,
    },
    "selective": {
        "tau_q": [round(value / 100, 2) for value in range(50, 100, 5)] + [0.99, 0.995, 0.999],
        "tau_m": [round(value / 100, 2) for value in range(0, 51, 5)],
        "selection": "maximize AutoRecovery subject to FalseAutoLink<=0.03 and WrongBranchAuto<=0.03",
    },
    "f1_minimum": {"development_positive": 150, "confirm_positive": 150, "development_negative": 150, "confirm_negative": 150, "absolute_stop": 75},
    "f2_gate": {"candidate_recall": 0.85},
    "f3_gate": {"candidate_recall": 0.85, "auto_recovery": 0.60, "false_auto_link": 0.03, "wrong_branch_auto": 0.03},
    "locks": {"expert": True, "confirm": True, "path": True, "relation_seeds_42_43": True, "new_architecture": True, "dense_training": True},
    "claim_boundary": "transductive dense evidence with section-disjoint heldout-annotator natural-gap evaluation; not unseen-image generalization",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)

