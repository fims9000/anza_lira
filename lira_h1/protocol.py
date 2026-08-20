"""Frozen protocol constants for ANZA-LIRA correctness hotfix H1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lira_final.protocol import HELDOUT_ANNOTATORS, TRAIN_ANNOTATORS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/lira_h1"
PARENT_DENSE_CACHE = ROOT / "results/lira_final/f1_gap_audit/dense_cache"
DEVELOPMENT_DENSE_CACHE = RESULT_ROOT / "development_dense_cache"
GAP_LENGTHS = (8, 12, 16, 24, 32, 40)
CUT_RADII = (3, 5, 7, 9, 11, 13, 15)
SPLIT_RANGES = {
    "h1_train": (1, 200),
    "h1_calibration": (203, 260),
    "h1_bug_audit": (263, 344),
    "h1_development": (347, 372),
    "h1_buffer": (373, 374),
    "h1_confirm": (375, 400),
}

PROTOCOL = {
    "version": "ANZA_LIRA_FINAL_CORRECTNESS_HOTFIX_H1",
    "parent_status": "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE",
    "authorized_now": ["H0_RIBBON", "H1_FREEZE", "H2_BUG_AUDIT_AND_CANDIDATE"],
    "splits": {name: list(bounds) for name, bounds in SPLIT_RANGES.items()},
    "trace_identity": "ordered non-junction, non-loop, non-border-truncated local segment within one annotator raster",
    "annotators": {"train": list(TRAIN_ANNOTATORS), "evaluation": list(HELDOUT_ANNOTATORS)},
    "placement": {
        "gap_lengths_px": list(GAP_LENGTHS),
        "minimum_context_px_each_side": 12,
        "minimum_supported_context_points_each_side": 8,
        "image_border_margin_px": 8,
        "evaluation_interventions_per_trace": 1,
        "seed_namespace": "h1_flat_cap_independent_placements",
    },
    "treatment": {
        "primitive": "flat_cap_trace_ribbon_exact_segment_projection",
        "validation_threshold": 0.12,
        "connectivity": 8,
        "anchor_points_each_side": 8,
        "anchor_raster_radius_px": 1,
        "candidate_radii_px": list(CUT_RADII),
        "roi_expansion_px": 17,
        "collateral_trace_radius_px": 2,
        "maximum_collateral_fraction": 0.05,
        "minimum_bug_audit_retention": 0.50,
        "treatment_validity": 1.0,
    },
    "dense": {
        "source": "mean frozen T1 U-Net seeds 41/42/43",
        "hard_threshold": 0.30,
        "soft_thresholds": [0.12, 0.18, 0.24],
        "retrained": False,
    },
    "fresh_development": {"target_cases": 400, "absolute_floor": 250},
    "candidate": {
        "implementation": "exact frozen real SBPP V3-B",
        "k": 12,
        "landing_band_px": 12.0,
        "source_port_availability_gate": 0.90,
        "branch_candidate_recall_gate": 0.85,
        "repair_allowed": False,
    },
    "locks": {"p0": True, "path": True, "confirm_contents": True, "expert": True, "new_architecture": True, "sbpp_modification": True},
    "claim_boundary": "controlled flat-cap evidence interruption on local crowd trace segments; not natural gaps or geological fault instances",
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_hash() -> str:
    return canonical_hash(PROTOCOL)

