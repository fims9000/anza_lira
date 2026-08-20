"""Fail-closed validator for the bounded ANZA-KIR IR0--IR2 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import KIR_VARIANTS
from .protocol import protocol_hash
from .runner import FREEZE, RESULT, k2_source_sha, source_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict[str, Any]:
    required = [FREEZE / "protocol.json", FREEZE / "benchmark_manifest.json", FREEZE / "source_manifest.json", FREEZE / "pre_ir2_receipt.json", RESULT / "ir0_forensic.json", RESULT / "metrics.json", RESULT / "ANZA_KIR_IR2_REPORT.md", RESULT / "raw_per_scene.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise ValueError(f"missing ANZA-KIR artifacts: {missing}")
    value = json.loads((FREEZE / "protocol.json").read_text()); benchmark = json.loads((FREEZE / "benchmark_manifest.json").read_text()); receipt = json.loads((FREEZE / "pre_ir2_receipt.json").read_text()); metrics = json.loads((RESULT / "metrics.json").read_text()); ir0 = json.loads((RESULT / "ir0_forensic.json").read_text())
    checks = {
        "parent_stop_preserved": metrics["parent_status"] == "STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED" and ir0["parent_status"] == "STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED",
        "protocol_hash": metrics["protocol_sha256"] == receipt["protocol_sha256"] == protocol_hash(value),
        "source_hash": receipt["source_sha256"] == source_manifest()["sha256"] == json.loads((FREEZE / "source_manifest.json").read_text())["sha256"],
        "old_k2_immutable": receipt["old_k2_source_sha256"] == ir0["old_k2_source_sha256"] == k2_source_sha(),
        "base_frozen": receipt["base_checkpoint_sha256"] == metrics["base_checkpoint_sha256"],
        "feature_norm_frozen": receipt["feature_norm_sha256"] == metrics["feature_norm_sha256"] == _sha(FREEZE / "feature_norm.json"),
        "benchmark_hash": receipt["benchmark_manifest_sha256"] == metrics["benchmark_manifest_sha256"] == benchmark["manifest_sha256"],
        "pool_minimum": benchmark["candidate_total"] >= 50_000,
        "dev_hard_minimum": benchmark["pools"]["mine-dev"]["selected_count"] >= 2000,
        "base_pair_error_valid": benchmark["base_pair_error_valid"] and 0.10 <= benchmark["pools"]["mine-dev"]["selected_pair_error"] <= 0.40,
        "matrix_complete": set(metrics["variants"]) == set(KIR_VARIANTS),
        "trainable_capacity_equal": len({metrics["variants"][variant]["run"]["trainable_parameter_count"] for variant in KIR_VARIANTS}) == 1,
        "scene_bootstrap": all(block[metric]["resamples"] == 10_000 and block[metric]["unit"] == "independent_scene" for block in metrics["bootstraps"].values() for metric in block),
        "locks_closed": not any(metrics[key] for key in ("confirm_opened", "cracks_accessed", "expert_accessed", "seeds_42_43_opened", "controlled_unfreezing_opened")) and not benchmark["confirm_evaluated"],
        "status_allowed": metrics["status"] in {"ANZA_KIR_RESIDUAL_PASS", "ANZA_KIR_SYMBOLIC_PASS_ANOSOV_UNRESOLVED", "STOP_ANZA_LOCAL_SYMBOLIC_ARCHITECTURE"},
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError(f"ANZA-KIR validation failed: {failed}")
    return {"validator": "PASS", "research_status": metrics["status"], "checks": checks, "confirm_opened": False, "cracks_accessed": False, "expert_accessed": False}
