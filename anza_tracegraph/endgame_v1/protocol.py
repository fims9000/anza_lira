"""Immutable TRACEGRAPH_P0_ENDGAME_V1 E1--E3 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/anza_tracegraph/endgame_v1"
E1_RESULT = RESULT_ROOT / "e1_p0"
E3_RESULT = RESULT_ROOT / "e3_relation"

PROTOCOL: dict[str, Any] = {
    "version": "TRACEGRAPH_P0_ENDGAME_V1",
    "authorized_phases": ["E1", "E2", "E3"],
    "parent_status": "SBPP_V3_B_BRANCH_COVERAGE_PASS",
    "dense_checkpoint_sha256": "95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6",
    "sbpp": {
        "version": "TRACEGRAPH_SBPP_V3_B_SOFT_SUPPORT",
        "tau_h": 0.35,
        "tau_s": 0.20,
        "candidate_k": 12,
        "curvature_split_radians": 0.70,
        "virtual_landing_band_px": 12.0,
    },
    "splits": {
        "relation_train": {"size": 19_200, "seed": 5_241_000_000},
        "relation_calibration": {"size": 3_840, "seed": 5_251_000_000},
        "relation_development": {"size": 3_840, "seed": 5_261_000_000},
    },
    "corridor": {
        "height": 32,
        "width": 64,
        "cross_extent_px": 16.0,
        "longitudinal_padding_px": 12.0,
        "channels": ["seismic", "gradient_x", "gradient_y", "visible_probability", "endpoint_markers", "candidate_corridor"],
        "source_history_px_min": 8.0,
        "destination_history_px_min": 8.0,
    },
    "p0": {
        "class": "path_completion.pair_classifier.EndpointPairClassifier",
        "source_sampling_unit": "source_case",
        "positive_wrong_candidates_max": 3,
        "none_candidates_max": 4,
        "wrong_order": "lowest frozen geometric score",
        "seed": 41,
        "epochs": 20,
        "batch_sources": 64,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "ranking_weight": 0.5,
        "ranking_temperature": 0.2,
        "early_stopping": False,
        "checkpoint_epoch": 20,
    },
    "selector": {
        "rule": "accept max candidate iff score >= one global threshold",
        "selection": "maximize RelationRecovery subject to FalseBridge <=0.02 and WrongBranch <=0.03",
        "threshold_source": "relation_calibration only",
        "false_bridge_max": 0.02,
        "wrong_branch_max": 0.03,
    },
    "development_gates": {
        "CCR_min": 0.87,
        "RelationRecovery_min": 0.84,
        "FalseBridge_max": 0.02,
        "WrongBranch_max": 0.03,
        "NONERecall_min": 0.90,
    },
    "bootstrap": {"unit": "source_scene", "resamples": 10_000, "seed": 41_003},
    "locks": {
        "transformer": True,
        "anza_change": True,
        "candidate_repair": True,
        "path": True,
        "confirm_metrics": True,
        "cracks": True,
        "expert": True,
        "seeds_42_43": True,
    },
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)
