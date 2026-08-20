"""Independent artifact-level validation of the frozen E1--E3 result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from anza_tracegraph.frozen_source import DENSE_CHECKPOINT

from ..p0.legacy_loader import SOURCE as P0_SOURCE
from ..protocol import E1_RESULT, E3_RESULT, PROTOCOL, ROOT, protocol_hash
from ..selector.metrics import relation_metrics


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict[str, Any]:
    metrics = json.loads((E3_RESULT / "metrics.json").read_text())
    split_manifest = json.loads((E3_RESULT / "split_manifest.json").read_text())
    calibration = json.loads((E3_RESULT / "calibration_metrics.json").read_text())
    checks: dict[str, bool] = {
        "protocol_hash": metrics["protocol_sha256"] == protocol_hash(),
        "dense_checkpoint_immutable": _sha(DENSE_CHECKPOINT) == PROTOCOL["dense_checkpoint_sha256"],
        "exact_p0_source": (E1_RESULT / "p0_source_sha256.txt").read_text().strip() == _sha(P0_SOURCE),
        "split_sizes": all(split_manifest[name]["size"] == PROTOCOL["splits"][name]["size"] for name in PROTOCOL["splits"]),
        "split_seeds_disjoint": len({split_manifest[name]["seed"] for name in PROTOCOL["splits"]}) == 3,
        "confirm_hash_only": split_manifest["old_v2_confirm"]["hash_only"] and not split_manifest["old_v2_confirm"]["inference_opened"] and not split_manifest["old_v2_confirm"]["metrics_opened"],
        "single_calibration_threshold": calibration["selected"] is not None and json.loads((E3_RESULT / "selector_freeze.json").read_text())["threshold_count"] == 1,
        "calibration_constraints": calibration["selected"] is not None and calibration["selected"]["FalseBridge"] <= 0.02 and calibration["selected"]["WrongBranch"] <= 0.03,
        "development_opened_after_freeze": json.loads((E3_RESULT / "selector_freeze.json").read_text())["development_opened"],
        "path_locked": not metrics["path_opened"],
        "confirm_locked": not metrics["confirm_opened"],
        "cracks_locked": not metrics["cracks_accessed"],
        "expert_locked": not metrics["expert_accessed"],
        "transformer_locked": not metrics["transformer_built"],
    }
    if metrics["development"] is not None:
        rows = _rows(E3_RESULT / "development_per_source.csv")
        integer_fields = ("positive", "accepted", "selected_none", "top_correct", "correct_accepted", "wrong_branch", "false_bridge", "candidate_miss_accepted")
        typed = []
        for row in rows:
            typed.append({**row, **{key: int(row[key]) for key in integer_fields}})
        recomputed = relation_metrics(typed)
        checks["development_source_count"] = len(rows) == PROTOCOL["splits"]["relation_development"]["size"]
        checks["metrics_recomputed"] = all(abs(float(recomputed[key]) - float(metrics["development"][key])) < 1e-12 for key in ("CCR", "RelationRecovery", "FalseBridge", "WrongBranch", "NONERecall"))
        gates = PROTOCOL["development_gates"]
        recomputed_pass = recomputed["CCR"] >= gates["CCR_min"] and recomputed["RelationRecovery"] >= gates["RelationRecovery_min"] and recomputed["FalseBridge"] <= gates["FalseBridge_max"] and recomputed["WrongBranch"] <= gates["WrongBranch_max"] and recomputed["NONERecall"] >= gates["NONERecall_min"]
        checks["status_matches_gates"] = (metrics["status"] == "P0_RELATION_SELECTOR_PASS") == recomputed_pass
        miss_sources = [row for row in rows if row["status"] == "CANDIDATE_MISS"]
        checks["candidate_misses_in_rr"] = int(recomputed["positive_sources"]) == sum(int(row["positive"]) for row in typed) and bool(miss_sources)
    validator = {"status": "PASS" if all(checks.values()) else "FAIL", "research_status": metrics["status"], "checks": checks}
    (E3_RESULT / "validator.json").write_text(json.dumps(validator, indent=2, sort_keys=True) + "\n")
    if validator["status"] != "PASS":
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return validator
