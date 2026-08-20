"""Fail-closed TG2 validator."""

from __future__ import annotations

import json
from typing import Any

from .models import VARIANTS
from .protocol import protocol_hash
from .runner import RESULT, source_manifest


def validate() -> dict[str, Any]:
    base_required = ["protocol.json", "protocol_hash.txt", "split_manifest.json", "candidate_recall.json", "candidate_per_case.csv", "source_manifest.json", "pretraining_receipt.json", "metrics.json", "ANZA_TRACEGRAPH_TG2_REPORT.md"]
    metrics_path = RESULT / "metrics.json"
    early_stop = metrics_path.exists() and json.loads(metrics_path.read_text()).get("status") == "STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK"
    required = base_required if early_stop else base_required + ["calibration.json", "per_source.csv", "per_pair.csv", "per_scene.csv", "operating_curves.csv", "bootstrap.json", "anza_bias_diagnostics.json"]
    missing = [name for name in required if not (RESULT / name).exists()]
    if missing: raise ValueError(f"missing TraceGraph artifacts: {missing}")
    protocol = json.loads((RESULT / "protocol.json").read_text()); splits = json.loads((RESULT / "split_manifest.json").read_text()); candidates = json.loads((RESULT / "candidate_recall.json").read_text()); receipt = json.loads((RESULT / "pretraining_receipt.json").read_text()); metrics = json.loads((RESULT / "metrics.json").read_text()); source = json.loads((RESULT / "source_manifest.json").read_text())
    checks = {"protocol_hash": metrics["protocol_sha256"] == receipt["protocol_sha256"] == protocol_hash() and (RESULT / "protocol_hash.txt").read_text().strip() == protocol_hash(), "source_hash": metrics["source_sha256"] == receipt["source_sha256"] == source["sha256"] == source_manifest()["sha256"], "split_hash": metrics["split_manifest_sha256"] == receipt["split_manifest_sha256"] == splits["manifest_sha256"], "confirm_hash_only": not splits["confirm_evaluated"], "sample_size": candidates["development_sources"] >= 2000 and candidates["positive_sources"] >= 1000 and candidates["none_sources"] >= 1000 and candidates["x_parallel_hard_sources"] >= 500, "locks": not any(metrics[key] for key in ("confirm_opened", "tg3_opened", "cracks_accessed", "expert_accessed", "seeds_42_43_opened", "p1g_opened"))}
    if early_stop:
        checks.update({"candidate_gate_failed": candidates["candidate_recall"] < 0.90, "training_locked": not metrics["training_opened"] and receipt["status"] == "TG2_TRAINING_LOCKED_BY_TG1", "status": metrics["status"] == "STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK"})
    else:
        checks.update({"candidate_recall": candidates["candidate_recall"] >= 0.90, "matrix": set(metrics["variants"]) == set(VARIANTS), "epochs": all(metrics["runs"][variant]["epoch"] == 20 for variant in VARIANTS), "p1_p2_capacity": metrics["runs"]["P2_anza_tracegraph"]["parameter_count"] == metrics["runs"]["P1_tracegraph"]["parameter_count"] + 1, "bootstrap": all(item[metric]["resamples"] == 10_000 and item[metric]["unit"] == "source endpoint / independent scene" for item in metrics["bootstraps"].values() for metric in item), "status": metrics["status"] in {"STOP_TRACEGRAPH_RELATION_NO_ARCHITECTURE_GAIN", "TRACEGRAPH_PASS_ANZA_BIAS_NOT_INCREMENTAL", "ANZA_TRACEGRAPH_CAUSAL_PASS"}})
    failed = [key for key, passed in checks.items() if not passed]
    if failed: raise ValueError(f"TraceGraph TG2 validation failed: {failed}")
    return {"validator": "PASS", "research_status": metrics["status"], "checks": checks, "confirm_opened": False, "tg3_opened": False, "cracks_accessed": False, "expert_accessed": False}
