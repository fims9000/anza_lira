"""End-to-end orchestrator for authorized phases E1--E3 only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from anza_tracegraph.frozen_source import DENSE_CHECKPOINT, load_frozen_source

from .p0.dataset import materialize_split
from .p0.legacy_loader import SOURCE as P0_SOURCE, architecture_receipt
from .p0.train import load_trained_p0, train_p0
from .protocol import E1_RESULT, E3_RESULT, PROTOCOL, RESULT_ROOT, ROOT, canonical_hash, protocol_hash
from .selector.calibration import calibrate_threshold, calibration_curve
from .selector.diagnostics import failure_attribution
from .selector.evaluator import score_split, write_csv
from .selector.metrics import bootstrap_source_metrics, relation_metrics, secondary_pair_metrics, source_decisions
from .split_data import SPLIT_SETTINGS, assert_seed_hygiene


CACHE = RESULT_ROOT / "cache"
PARENT = ROOT / "results/anza_tracegraph/sbpp_v3_b"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> dict[str, Any]:
    paths = [
        ROOT / path
        for path in (
            "anza_tracegraph/endgame_v1/protocol.py",
            "anza_tracegraph/endgame_v1/split_data.py",
            "anza_tracegraph/endgame_v1/p0/legacy_loader.py",
            "anza_tracegraph/endgame_v1/p0/corridor.py",
            "anza_tracegraph/endgame_v1/p0/dataset.py",
            "anza_tracegraph/endgame_v1/p0/train.py",
            "anza_tracegraph/endgame_v1/selector/calibration.py",
            "anza_tracegraph/endgame_v1/selector/metrics.py",
            "anza_tracegraph/endgame_v1/selector/evaluator.py",
            "anza_tracegraph/endgame_v1/selector/diagnostics.py",
            "anza_tracegraph/endgame_v1/runner.py",
            "anza_tracegraph/endgame_v1/validators/e3.py",
            "scripts/run_tracegraph_endgame_v1_e1_e3.py",
            "scripts/validate_tracegraph_endgame_v1_e1_e3.py",
        )
    ]
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]
    return {"files": rows, "sha256": canonical_hash(rows)}


def _copy_training_rows() -> None:
    shutil.copy2(CACHE / "relation_train_sources.csv", E1_RESULT / "train_sources.csv")
    shutil.copy2(CACHE / "relation_train_candidates.csv", E1_RESULT / "train_pairs.csv")


def _per_stratum(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for stratum in sorted({str(row["stratum"]) for row in decisions}):
        local = [row for row in decisions if row["stratum"] == stratum]
        metrics = relation_metrics(local)
        row: dict[str, Any] = {
            "stratum": stratum,
            "sources": len(local),
            "positive_sources": metrics["positive_sources"],
            "candidate_available_positives": metrics["candidate_available_positives"],
            "candidate_miss_positives": metrics["candidate_miss_positives"],
        }
        if metrics["positive_sources"]:
            row.update({"candidate_availability": metrics["candidate_available_positives"] / metrics["positive_sources"], "CCR": metrics["CCR"], "RelationRecovery": metrics["RelationRecovery"], "WrongBranch": metrics["WrongBranch"]})
        else:
            row.update({"FalseBridge": metrics["FalseBridge"], "NONERecall": metrics["NONERecall"]})
        output.append(row)
    return output


def _write_report(metrics: dict[str, Any], calibration: dict[str, Any], per_stratum: list[dict[str, Any]]) -> None:
    selected = calibration["selected"]
    lines = [
        "# TRACEGRAPH P0 ENDGAME V1 — E1 to E3",
        "",
        f"Status: `{metrics['status']}`",
        "",
        "Exact historical five-convolution corridor P0 was retrained on fresh source-disjoint relation streams. SBPP V3-B remained frozen at `tau_s=0.20`, `K=12`; path, confirm, CRACKS, expert, Transformer, and ANZA changes remained locked.",
        "",
        "## Calibration",
        "",
        f"- selected threshold: `{selected['threshold']:.9f}`",
        f"- calibration RR / FB / WB: `{selected['RelationRecovery']:.6f} / {selected['FalseBridge']:.6f} / {selected['WrongBranch']:.6f}`",
        "- selection rule: maximize all-positive RelationRecovery subject to `FB<=0.02` and `WB<=0.03`.",
        "",
        "## Fresh relation development",
        "",
        f"- CCR: `{metrics['development']['CCR']:.6f}` (gate >=0.87)",
        f"- RelationRecovery: `{metrics['development']['RelationRecovery']:.6f}` (gate >=0.84)",
        f"- FalseBridge: `{metrics['development']['FalseBridge']:.6f}` (gate <=0.02)",
        f"- WrongBranch: `{metrics['development']['WrongBranch']:.6f}` (gate <=0.03)",
        f"- NONE recall: `{metrics['development']['NONERecall']:.6f}` (gate >=0.90)",
        f"- candidate availability: `{metrics['development']['candidate_available_positives']}/{metrics['development']['positive_sources']}`",
        "",
        "## Secondary diagnostics",
        "",
    ]
    for key, value in metrics["secondary"].items():
        lines.append(f"- {key}: `{value:.6f}`")
    weak = next(row for row in per_stratum if row["stratum"] == "weak_branch_continue")
    attribution = metrics["failure_attribution"]
    lines += [
        "",
        "## Weak branch boundary",
        "",
        f"- candidate availability: `{weak.get('candidate_availability', float('nan')):.6f}`",
        f"- candidate-conditional CCR: `{weak.get('CCR', float('nan')):.6f}`",
        f"- all-source RR: `{weak.get('RelationRecovery', float('nan')):.6f}`",
        "",
        "No weak-branch system-success claim is made unless its all-source RR reaches 0.70.",
        "",
        "## Frozen failure attribution",
        "",
        f"- bottleneck: `{attribution['bottleneck']}`",
        f"- best development RR under `FB<=0.02, WB<=0.03`: `{attribution['best_development_relation_recovery_under_frozen_safety_constraints']:.6f}`",
        f"- minimum development FB at `CCR>=0.87`: `{attribution['minimum_false_bridge_observed_at_CCR_at_least_0_87']:.6f}`",
        f"- accepted NONE sources at the frozen threshold: `{attribution['accepted_none_count_at_frozen_threshold']}`; by stratum: `{json.dumps(attribution['accepted_none_by_stratum'], sort_keys=True)}`",
        "",
        "This is a post-gate diagnostic only. It does not reopen calibration, change the selected threshold, or authorize a repair in this cycle.",
        "",
        "## Decision",
        "",
        "E4 widest path is authorized only when status is `P0_RELATION_SELECTOR_PASS`. This run stops before E4 in every case.",
    ]
    (E3_RESULT / "TRACEGRAPH_P0_RELATION_REPORT.md").write_text("\n".join(lines) + "\n")


def run(*, device: str = "cuda") -> dict[str, Any]:
    completed = E3_RESULT / "metrics.json"
    if completed.exists():
        return json.loads(completed.read_text())
    assert_seed_hygiene()
    parent_metrics = json.loads((PARENT / "metrics.json").read_text())
    parent_freeze = json.loads((PARENT / "sbpp_v3_b_freeze.json").read_text())
    if parent_metrics.get("status") != "SBPP_V3_B_BRANCH_COVERAGE_PASS" or float(parent_freeze.get("selected_tau_s")) != 0.20:
        raise PermissionError("frozen SBPP V3-B parent drift")
    if _sha(DENSE_CHECKPOINT) != PROTOCOL["dense_checkpoint_sha256"]:
        raise PermissionError("frozen dense checkpoint drift")
    E1_RESULT.mkdir(parents=True, exist_ok=True)
    E3_RESULT.mkdir(parents=True, exist_ok=True)
    _json(E1_RESULT / "protocol.json", PROTOCOL)
    _json(E3_RESULT / "protocol.json", PROTOCOL)
    (E1_RESULT / "p0_source_file.txt").write_text(str(P0_SOURCE.relative_to(ROOT)) + "\n")
    (E1_RESULT / "p0_source_sha256.txt").write_text(_sha(P0_SOURCE) + "\n")
    _json(E1_RESULT / "p0_architecture.json", architecture_receipt())
    model = load_frozen_source(device)
    manifests = {}
    for split in ("relation_train", "relation_calibration", "relation_development"):
        manifests[split] = materialize_split(split, model=model, device=device, output_dir=CACHE)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parent_split = json.loads((PARENT / "split_manifest.json").read_text())
    split_manifest = {
        split: {**manifest, "hash_frozen_before_training": True, "relation_scores_opened": False}
        for split, manifest in manifests.items()
    }
    split_manifest["old_v2_confirm"] = {
        "sha256": parent_split["confirm"]["sha256"],
        "hash_only": True,
        "inference_opened": False,
        "metrics_opened": False,
    }
    _json(E1_RESULT / "split_manifest.json", split_manifest)
    _copy_training_rows()
    training = train_p0(CACHE, E1_RESULT, device=device)
    (E1_RESULT / "checkpoint_sha256.txt").write_text(training["checkpoint_sha256"] + "\n")
    checkpoint = E1_RESULT / "checkpoint.pt"
    p0 = load_trained_p0(checkpoint, device=device)
    calibration_sources, calibration_candidates = score_split(p0, CACHE, "relation_calibration", device=device)
    split_manifest["relation_calibration"]["relation_scores_opened"] = True
    calibration = calibrate_threshold(calibration_sources, calibration_candidates)
    write_csv(E3_RESULT / "calibration_curve.csv", calibration["curve"])
    calibration_decisions = [] if calibration["selected"] is None else source_decisions(calibration_sources, calibration_candidates, float(calibration["selected"]["threshold"]))
    calibration_payload = {
        "status": calibration["status"],
        "selected": calibration["selected"],
        "secondary": secondary_pair_metrics(calibration_candidates, calibration_decisions) if calibration_decisions else None,
        "development_opened": False,
    }
    _json(E3_RESULT / "calibration_metrics.json", calibration_payload)
    if calibration["selected"] is None:
        metrics = {"status": "STOP_P0_OPERATING_POINT_INFEASIBLE", "protocol_sha256": protocol_hash(), "development": None, "locks": PROTOCOL["locks"]}
        _json(E3_RESULT / "metrics.json", metrics)
        _json(E3_RESULT / "split_manifest.json", split_manifest)
        (E3_RESULT / "TRACEGRAPH_P0_RELATION_REPORT.md").write_text("# TRACEGRAPH P0 ENDGAME V1\n\nStatus: `STOP_P0_OPERATING_POINT_INFEASIBLE`\n\nNo relation-development scores, path, confirm, CRACKS, expert, Transformer, or ANZA changes were opened.\n")
        return metrics
    threshold = float(calibration["selected"]["threshold"])
    selector_freeze = {
        "checkpoint_sha256": training["checkpoint_sha256"],
        "threshold": threshold,
        "threshold_count": 1,
        "selection_split": "relation_calibration",
        "selection_rule": PROTOCOL["selector"]["selection"],
        "sbpp_protocol_sha256": _sha(PARENT / "protocol.json"),
        "sbpp_freeze_sha256": _sha(PARENT / "sbpp_v3_b_freeze.json"),
        "corridor_code_sha256": _sha(ROOT / "anza_tracegraph/endgame_v1/p0/corridor.py"),
        "metric_definitions": {
            "CCR": "correct accepted / candidate-available positives",
            "RelationRecovery": "correct accepted / all positives including candidate misses",
            "FalseBridge": "accepted / NO_VALID_CONTINUATION",
            "WrongBranch": "accepted wrong / candidate-available positives",
        },
        "development_gates": PROTOCOL["development_gates"],
        "development_opened": False,
        "path_opened": False,
        "confirm_opened": False,
    }
    _json(E3_RESULT / "selector_freeze.json", selector_freeze)
    development_sources, development_candidates = score_split(p0, CACHE, "relation_development", device=device)
    split_manifest["relation_development"]["relation_scores_opened"] = True
    selector_freeze["development_opened"] = True
    _json(E3_RESULT / "selector_freeze.json", selector_freeze)
    decisions = source_decisions(development_sources, development_candidates, threshold)
    development = relation_metrics(decisions)
    secondary = secondary_pair_metrics(development_candidates, decisions)
    per_stratum = _per_stratum(decisions)
    bootstrap = bootstrap_source_metrics(decisions, resamples=int(PROTOCOL["bootstrap"]["resamples"]), seed=int(PROTOCOL["bootstrap"]["seed"]))
    operating = calibration_curve(development_sources, development_candidates)
    attribution = failure_attribution(decisions, operating)
    gates = PROTOCOL["development_gates"]
    checks = {
        "CCR": development["CCR"] >= gates["CCR_min"],
        "RelationRecovery": development["RelationRecovery"] >= gates["RelationRecovery_min"],
        "FalseBridge": development["FalseBridge"] <= gates["FalseBridge_max"],
        "WrongBranch": development["WrongBranch"] <= gates["WrongBranch_max"],
        "NONERecall": development["NONERecall"] >= gates["NONERecall_min"],
    }
    status = "P0_RELATION_SELECTOR_PASS" if all(checks.values()) else "STOP_P0_RELATION_SELECTOR"
    metrics = {
        "status": status,
        "protocol_sha256": protocol_hash(),
        "threshold": threshold,
        "development": development,
        "secondary": secondary,
        "failure_attribution": attribution,
        "gates": checks,
        "locks": PROTOCOL["locks"],
        "path_authorized": status == "P0_RELATION_SELECTOR_PASS",
        "path_opened": False,
        "confirm_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
        "transformer_built": False,
    }
    write_csv(E3_RESULT / "development_per_source.csv", decisions)
    write_csv(E3_RESULT / "development_per_candidate.csv", development_candidates)
    write_csv(E3_RESULT / "development_per_stratum.csv", per_stratum)
    write_csv(E3_RESULT / "operating_curve.csv", operating)
    _json(E3_RESULT / "bootstrap.json", bootstrap)
    _json(E3_RESULT / "failure_attribution.json", attribution)
    _json(E3_RESULT / "metrics.json", metrics)
    _json(E3_RESULT / "split_manifest.json", split_manifest)
    _json(E3_RESULT / "source_manifest.json", _source_manifest())
    (E3_RESULT / "checkpoint_sha256.txt").write_text(training["checkpoint_sha256"] + "\n")
    _write_report(metrics, calibration, per_stratum)
    return metrics
