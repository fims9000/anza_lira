"""Frozen CRACKS Intervention Endgame V1 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lira_final.protocol import HELDOUT_ANNOTATORS, TRAIN_ANNOTATORS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/lira_intervention_final"
DENSE_CACHE = ROOT / "results/lira_final/f1_gap_audit/dense_cache"
GAP_LENGTHS = (8, 12, 16, 24, 32, 40)

# Frozen before any intervention was generated or scored.  347--400 remains the
# unopened confirm range from the preceding natural-gap protocol.
SPLIT_RANGES = {
    "ig_train": (1, 200),
    "ig_calibration": (203, 246),
    "ig_development": (249, 344),
    "ig_confirm": (347, 400),
}

PROTOCOL = {
    "version": "ANZA_LIRA_CRACKS_INTERVENTION_ENDGAME_V1",
    "parent_stop": "STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT",
    "authorized": ["I0_FREEZE", "I1_BENCHMARK", "I2_SBPP", "I3_P0_SEED41"],
    "splits": {key: list(value) for key, value in SPLIT_RANGES.items()},
    "split_buffer_sections": 2,
    "annotators": {
        "train": list(TRAIN_ANNOTATORS),
        "evaluation": list(HELDOUT_ANNOTATORS),
    },
    "trace_identity": "ordered non-junction, non-loop, non-border-truncated local raster skeleton segment",
    "intervention": {
        "gap_lengths_px": list(GAP_LENGTHS),
        "minimum_visible_context_px_each_side": 12,
        "minimum_endpoint_clearance_px": 8,
        "dense_evidence_tube_radius_px": 3,
        "image_changed": False,
        "evaluation_interventions_per_trace": 1,
        "training_interventions_per_trace_max": 2,
    },
    "benchmark_minimum": {"calibration": 750, "development": 750, "confirm": 750},
    "benchmark_target": {"calibration": 1000, "development": 1500, "confirm": 1500},
    "dense": {
        "source": "mean frozen T1 U-Net seeds 41/42/43",
        "tau_h": 0.30,
        "soft_thresholds": [0.12, 0.18, 0.24],
        "retrained": False,
    },
    "candidate": {"implementation": "frozen SBPP V3-B", "k": 12, "landing_band_px": 12.0},
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
        "maximum_negatives": 3,
        "maximum_training_sources": 4096,
        "augmentation": "four deterministic axial/cross-axis flips",
    },
    "selective": {
        "tau_q": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999],
        "tau_m": [round(value / 100, 2) for value in range(0, 51, 5)],
        "selection": "maximize AutoRecovery under FalseAutoLink<=0.03 and WrongBranchAuto<=0.03",
    },
    "i2_gate": {"candidate_recall_at_12": 0.90},
    "i3_gate": {"candidate_recall_at_12": 0.90, "top1": 0.90, "auto_recovery": 0.60, "false_auto_link": 0.03, "wrong_branch_auto": 0.03},
    "locks": {"confirm_contents": True, "expert": True, "path": True, "seeds_42_43": True, "new_architecture": True},
    "claim_boundary": "section-disjoint controlled evidence-only gaps on real CRACKS images and crowd traces; not natural-gap or unseen-image generalization",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)

