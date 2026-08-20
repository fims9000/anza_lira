"""Frozen TG0--TG2 protocol."""

from __future__ import annotations

import hashlib
import json


PROTOCOL = {
    "version": "ANZA_TRACEGRAPH_TG0_TG2_V1",
    "parent_status": "STOP_ANZA_LOCAL_SYMBOLIC_ARCHITECTURE",
    "dense_source": {
        "variant": "ANZA-KIR R0_static_residual",
        "checkpoint": "/home/lebedeffson/Code/_wip_backups/anza_lira/anza_kir_checkpoints/R0_static_residual-d7695ee995a7ec56.pt",
        "checkpoint_sha256": "95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6",
        "threshold": 0.35,
        "threshold_source": "frozen ANZA-KIR calibration-natural R0 threshold",
        "tracelets": "extracted from thresholded frozen prediction; relation corridor x=[35,50) removed before skeletonization",
    },
    "seed": 41,
    "splits": {"train": 4096, "calibration": 2048, "development": 2048, "confirm": 4096},
    "split_seeds": {"train": 4_101_000_000, "calibration": 4_111_000_000, "development": 4_121_000_000, "confirm": 4_131_000_000},
    "scene_types": [
        "straight", "curved", "s_curve", "x_crossing", "acute_crossing", "close_parallel",
        "parallel_gap_confuser", "weak_branch", "y_junction", "t_junction", "long_gap",
        "none", "multiple_plausible", "low_contrast", "cluttered_corridor", "partial_occlusion",
    ],
    "tracelets": {"threshold": 0.35, "min_length": 8, "tangent_points": 5, "curvature_split_radians": 0.70, "skeleton_connectivity": 8, "truth_match_tolerance_px": 6.0},
    "candidates": {"k_max": 8, "min_distance": 6.0, "max_distance": 68.0, "max_tangent_mismatch_degrees": 78.0, "score": "distance + 8*axial_tangent_error; no ANZA"},
    "corridor": {"height": 32, "width": 64, "cross_extent_px": 16.0, "longitudinal_padding_px": 12.0, "channels": 10, "visible_context_px_min": 12},
    "p0": {"architecture": "five-convolution corridor classifier, mean+max readout", "none_rule": "calibration-only binary threshold"},
    "p1_p2": {"layers": 2, "model_dim": 128, "heads": 4, "ffn_dim": 256, "dropout": 0.1, "none": True, "pair_candidates": 8},
    "anza_bias": {"application": "SRC query to corridor tokens only", "h": 0.35, "beta": "softplus trainable, raw init -4.0", "coordinate_normalization": "corridor half extents"},
    "training": {"epochs": 20, "batch_size": 32, "optimizer": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001},
    "bootstrap": {"resamples": 10_000, "unit": "source endpoint / independent scene", "seed": 4_141_000_041},
    "gates": {
        "candidate_recall": 0.90,
        "p1_vs_p0": "TPR@FPR05 +0.08 OR Top1+NONE +0.08 OR wrong-branch <=0.70, plus parallel safety <=+0.01",
        "p2_vs_p1": "TPR@FPR05 +0.05 OR Top1+NONE +0.05 OR wrong-branch <=0.80 with positive paired CI, plus parallel safety",
    },
    "locks": {"tg3_path": True, "confirm": True, "cracks": True, "expert": True, "seeds_42_43": True, "p1g": True},
}


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
