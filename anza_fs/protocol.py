"""Immutable pre-gradient protocol for ANZA-FS H3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hard_bench_v6 import hard_bench_config


ROOT = Path(__file__).resolve().parents[1]
H1_ROOT = ROOT / "results" / "anza_hs" / "h1"
H3_ROOT = ROOT / "results" / "anza_fs" / "h3"
PREGRADIENT_ROOT = H3_ROOT / "pre_gradient"
VERSION = "ANZA_FS_H3_V1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_payload() -> dict[str, Any]:
    parent = json.loads((H1_ROOT / "metrics.json").read_text())
    if parent.get("status") != "HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL":
        raise ValueError("frozen ANZA-HS H1 status changed")
    return {
        "version": VERSION,
        "parent_h1_metrics_sha256": digest(H1_ROOT / "metrics.json"),
        "parent_h1_report_sha256": digest(H1_ROOT / "ANZA_HS_H1_REPORT.md"),
        "parent_h1_status": parent["status"],
        "question": "Do distinct longitudinal propagation and transverse suppression responses reduce false bridges, and does reciprocal hyperbolic tying add value over a free five-lobe control?",
        "stressbench": hard_bench_config(),
        "matrix": ["F0_backbone", "F1_old_generic", "F2_free_foliation", "F3_anza_fs"],
        "architecture": {
            "M": 8,
            "support": 9,
            "base_scale": 1.5,
            "lambda": 0.35,
            "delta_u": "1.5*sigma_u",
            "delta_s": "1.5*sigma_s",
            "placements": ["decoder_1_4", "decoder_1_2"],
            "gamma_init": 0.0,
            "feature_tuple": ["C", "U-C", "C-S"],
        },
        "orientation_target": {"sigma_theta": 0.20, "background_weight": 0.25, "source": "visible generator branch axes only"},
        "training": {
            "seed": 41,
            "epochs": 15,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 8,
            "train_samples": 512,
            "segmentation_loss": "BCE_plus_softDice_plus_0.25_softclDice",
            "orientation_loss_weight": 0.10,
            "early_stopping": False,
        },
        "threshold": {
            "grid": [round(0.05 + 0.025 * index, 3) for index in range(37)],
            "calibration_split": "calibration[0:512]",
            "development_split": "development[0:512]",
            "primary_rule": "highest calibration threshold with mean branch recall >= 0.95",
            "matched_dice_rule": "F3 threshold closest to comparator calibration Dice at comparator primary threshold",
            "matched_precision_rule": "F3 threshold closest to comparator calibration precision at comparator primary threshold",
        },
        "gate": {
            "branch_recall_minimum": 0.95,
            "F3_vs_F1_fbr_ratio_max": 0.70,
            "F3_vs_F2_fbr_ratio_max": 0.80,
            "F3_vs_F2_fragmentation_ratio_max": 0.85,
            "dice_noninferiority": -0.005,
        },
        "bootstrap_resamples": 10000,
        "training_allowed_after_pregradient_pass": True,
        "confirm_opened": False,
        "test_created": False,
        "cracks_accessed": False,
        "expert_accessed": False,
        "H4_opened": False,
        "lambda_tuned": False,
        "M_tuned": False,
        "base_scale_tuned": False,
    }


def freeze_protocol(output_root: Path = PREGRADIENT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = output_root / "protocol.json"
    if path.exists() and path.read_text() != encoded:
        raise ValueError("ANZA-FS H3 protocol drift")
    path.write_text(encoded)
    (output_root / "protocol_hash.txt").write_text(canonical_hash(payload) + "\n")
    return payload
