"""Frozen ANZA-LIRA Graph-Cut Intervention V2 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lira_final.protocol import HELDOUT_ANNOTATORS, TRAIN_ANNOTATORS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/lira_graph_cut_v2"
DENSE_CACHE = ROOT / "results/lira_final/f1_gap_audit/dense_cache"
GAP_LENGTHS = (8, 12, 16, 24, 32, 40)
CUT_RADII = (3, 5, 7, 9, 11, 13, 15)

# Frozen before any V2 placement, manipulation-validity count, or SBPP score.
SPLIT_RANGES = {
    "gc_train": (1, 200),
    "gc_calibration": (203, 260),
    "gc_development": (263, 344),
    "gc_confirm": (347, 400),
}

PROTOCOL = {
    "version": "ANZA_LIRA_GRAPH_CUT_INTERVENTION_V2",
    "parent": "ANZA_LIRA_CRACKS_INTERVENTION_ENDGAME_V1",
    "parent_status": "STOP_LIRA_INTERVENTION_CANDIDATE",
    "authorized_now": ["FREEZE", "GRAPH_CUT_BENCHMARK", "FROZEN_SBPP_CANDIDATE"],
    "splits": {key: list(value) for key, value in SPLIT_RANGES.items()},
    "split_buffer_sections": 2,
    "annotators": {"train": list(TRAIN_ANNOTATORS), "evaluation": list(HELDOUT_ANNOTATORS)},
    "trace_identity": "ordered non-junction, non-loop, non-border-truncated local segment within one annotator raster",
    "placement": {
        "gap_lengths_px": list(GAP_LENGTHS),
        "minimum_context_px_each_side": 12,
        "minimum_supported_context_points_each_side": 8,
        "image_border_margin_px": 8,
        "evaluation_interventions_per_trace": 1,
        "seed_namespace": "graph_cut_v2_independent_placements",
    },
    "treatment": {
        "validation_threshold": 0.12,
        "connectivity": 8,
        "anchor_points_each_side": 8,
        "anchor_raster_radius_px": 1,
        "candidate_radii_px": list(CUT_RADII),
        "collateral_trace_radius_px": 2,
        "maximum_collateral_fraction": 0.05,
        "require_pre_cut_connection": True,
        "minimum_retention": 0.50,
        "treatment_validity": 1.0,
    },
    "dense": {
        "source": "mean frozen T1 U-Net seeds 41/42/43",
        "hard_threshold": 0.30,
        "soft_thresholds": [0.12, 0.18, 0.24],
        "retrained": False,
    },
    "minimum_valid_cases": {"calibration": 1000, "development": 1500, "absolute_development": 750, "confirm": 1500, "absolute_confirm": 750},
    "candidate": {
        "implementation": "exact frozen real SBPP V3-B",
        "k": 12,
        "landing_band_px": 12.0,
        "source_port_availability_gate": 0.95,
        "branch_candidate_recall_gate": 0.90,
        "repair_allowed": False,
    },
    "locks": {"p0": True, "path": True, "confirm_contents": True, "expert": True, "new_architecture": True, "sbpp_modification": True},
    "claim_boundary": "topology-valid controlled evidence cuts on section-disjoint real CRACKS images/crowd traces; not natural gaps or unseen-image generalization",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)

