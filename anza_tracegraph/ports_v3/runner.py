"""Zero-training TRACEGRAPH SBPP V3-A runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from anza_tracegraph.data_v2.generator import RELATION_CORRIDOR_X, generate_scene, split_hash
from anza_tracegraph.data_v2.strata import MAIN_SAFETY_STRATA, POSITIVE_STRATA, SPLIT_SEEDS, SPLIT_SIZES, STRATA
from anza_tracegraph.data_v2.validator import validate_generator
from anza_tracegraph.frozen_source import DENSE_CHECKPOINT, DENSE_THRESHOLD, infer_dense, load_frozen_source

from .branches import Branch, extract_branches
from .candidates import propose_branch_candidates, select_source_port, source_ports
from .metrics import branch_match, wilson_interval
from .micro_branches import micro_branches
from .valley_ports import confidence_valley_ports


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/anza_tracegraph/sbpp_v3_a"
TAU_CANDIDATES = (0.20, 0.25, 0.30, 0.35)
K_VALUES = (4, 8, 12, 16)
PROTOCOL: dict[str, Any] = {
    "version": "TRACEGRAPH_SBPP_V3_A",
    "parent": "CANDIDATE_AUDIT_V2_COMPLETE",
    "benchmark": "TRACEGRAPH_RELATION_V2",
    "dense_checkpoint_sha256": "95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6",
    "dense_threshold": 0.35,
    "relation_corridor_x": list(RELATION_CORRIDOR_X),
    "curvature_split_radians": 0.70,
    "ports": {"terminal": True, "junction_offset_px": 4.0, "virtual_band_px": 12.0, "virtual_step_px": 2.0, "valley_ratio": 0.80, "micro_length_px": [4.0, 8.0]},
    "tau_micro_candidates": list(TAU_CANDIDATES),
    "tau_selection": "smallest mean branch pool among calibration recall@12 >=0.97; ties prefer larger tau",
    "candidate": {"min_distance": 6.0, "max_distance": 68.0, "max_directed_angle_degrees": 78.0, "score": "distance+8*max(directed endpoint angles)", "dedup": "one candidate per destination branch", "primary_k": 12, "curve_k": list(K_VALUES)},
    "branch_match_evaluation_only": {"tube_radius_px": 3.0, "minimum_predicted_fraction": 0.60},
    "gates": {"branch_recall_at_12": 0.95, "median_candidate_branches": 8.0, "p95_candidate_branches": 16.0, "main_stratum_recall": 0.90},
    "locks": {"training": True, "p0": True, "transformer": True, "path": True, "confirm_evaluation": True, "cracks": True, "expert": True},
}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: path.write_text(""); return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_hash() -> str: return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest() -> dict[str, Any]:
    paths = [ROOT / path for path in ("anza_tracegraph/data_v2/generator.py", "anza_tracegraph/data_v2/strata.py", "anza_tracegraph/data_v2/validator.py", "anza_tracegraph/ports_v3/branches.py", "anza_tracegraph/ports_v3/curvature_split.py", "anza_tracegraph/ports_v3/terminal_ports.py", "anza_tracegraph/ports_v3/junction_ports.py", "anza_tracegraph/ports_v3/virtual_landing.py", "anza_tracegraph/ports_v3/valley_ports.py", "anza_tracegraph/ports_v3/micro_branches.py", "anza_tracegraph/ports_v3/candidates.py", "anza_tracegraph/ports_v3/metrics.py", "anza_tracegraph/ports_v3/runner.py", "anza_tracegraph/ports_v3/validator.py")]
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]; digest = hashlib.sha256()
    for row in rows: digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": rows, "sha256": digest.hexdigest()}


def _context(scene: dict[str, Any], probability: np.ndarray, tau_micro: float) -> dict[str, Any]:
    mask = np.asarray(probability) >= DENSE_THRESHOLD; start, end = scene["input"]["relation_corridor_x"]; mask[:, start:end] = False
    branches = extract_branches(mask, probability, scene["input"]["model_input"][0], tau_micro=tau_micro)
    ports = source_ports(branches, probability); source = select_source_port(ports, scene["input"]["source_query_yx"], scene["input"]["source_tangent_yx"])
    candidates = () if source is None else propose_branch_candidates(source, branches, probability)
    return {"mask": mask, "branches": branches, "ports": ports, "source": source, "candidates": candidates}


def _evaluate(scene: dict[str, Any], probability: np.ndarray, tau_micro: float) -> dict[str, Any]:
    context = _context(scene, probability, tau_micro); truth = scene["truth"]; positive = bool(truth["has_valid_continuation"]); target = truth["destination_branch"]
    matches: dict[int, bool] = {}; fractions: dict[int, float] = {}
    if positive and target is not None:
        for branch in context["branches"]:
            match, fraction, _ = branch_match(branch, target); matches[branch.branch_id] = match; fractions[branch.branch_id] = fraction
    candidates = context["candidates"]
    row: dict[str, Any] = {"split": scene["input"]["split"], "index": scene["input"]["index"], "stratum": scene["input"]["stratum"], "positive": int(positive), "source_available": int(context["source"] is not None), "branch_count": len(context["branches"]), "candidate_branch_count": len(candidates), "terminal_port_count": sum(port.port_type == "terminal" for port in context["ports"]), "junction_port_count": sum(port.port_type == "junction_arm" for port in context["ports"]), "valley_port_count": sum(port.port_type.startswith("valley") for port in context["ports"]), "micro_branch_count": len(micro_branches(context["branches"]))}
    for k in K_VALUES: row[f"branch_recalled_at_{k}"] = int(not positive or any(matches.get(candidate.destination_branch_id, False) for candidate in candidates[:k]))
    row["correct_branch_extracted"] = int(any(matches.values())) if positive else 0
    row["correct_branch_valid_landing"] = int(any(matches.get(candidate.destination_branch_id, False) for candidate in candidates)) if positive else 0
    row["correct_branch_best_rank"] = next((rank for rank, candidate in enumerate(candidates) if matches.get(candidate.destination_branch_id, False)), -1)
    row["endpoint_close_but_wrong_branch"] = 0
    if positive and target is not None:
        endpoint = np.asarray(target[0]); close_wrong = [candidate for candidate in candidates[:12] if np.linalg.norm(np.asarray(candidate.landing_point_yx) - endpoint) <= 6.0 and not matches.get(candidate.destination_branch_id, False)]
        row["endpoint_close_but_wrong_branch"] = int(bool(close_wrong) and not row["branch_recalled_at_12"])
    if positive and not row["branch_recalled_at_12"]:
        if target is None: category = "F3"
        elif row["correct_branch_valid_landing"] and row["correct_branch_best_rank"] >= 12: category = "B3"
        elif row["correct_branch_extracted"]:
            matching = [branch for branch in context["branches"] if matches.get(branch.branch_id, False)]
            special = any(branch.candidate_only or branch.start_type == "junction" or branch.end_type == "junction" for branch in matching)
            category = "C3" if special else "A3"
        else:
            pixels = np.rint(target).astype(int); pixels[:, 0] = np.clip(pixels[:, 0], 0, probability.shape[0] - 1); pixels[:, 1] = np.clip(pixels[:, 1], 0, probability.shape[1] - 1)
            soft_fraction = float(np.mean(probability[pixels[:, 0], pixels[:, 1]] >= min(TAU_CANDIDATES)))
            row["target_soft_support_fraction"] = soft_fraction; category = "D3" if soft_fraction >= 0.25 else "E3"
        row["miss_category"] = category
    else:
        row["miss_category"] = "NONE"
    row.setdefault("target_soft_support_fraction", "")
    return row


def _run_split(model: Any, split: str, tau_values: tuple[float, ...], *, device: str, batch_size: int = 64) -> dict[float, list[dict[str, Any]]]:
    rows = {tau: [] for tau in tau_values}
    for start in range(0, SPLIT_SIZES[split], batch_size):
        scenes = [generate_scene(split, index) for index in range(start, min(start + batch_size, SPLIT_SIZES[split]))]
        probabilities, _ = infer_dense(model, np.stack([scene["input"]["model_input"] for scene in scenes]), device=device)
        for scene, probability in zip(scenes, probabilities):
            for tau in tau_values: rows[tau].append(_evaluate(scene, probability, tau))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["positive"]]
    counts = np.asarray([row["candidate_branch_count"] for row in rows], dtype=float)
    successes = sum(row["branch_recalled_at_12"] for row in positives); interval = wilson_interval(successes, len(positives))
    return {"sources": len(rows), "positive_sources": len(positives), "none_sources": len(rows) - len(positives), "branch_recall_at_12": successes / len(positives), "branch_recall_at_12_wilson95": list(interval), "median_candidate_branches": float(np.median(counts)), "p95_candidate_branches": float(np.quantile(counts, 0.95)), "mean_candidate_branches": float(counts.mean()), "endpoint_close_but_wrong_branch": sum(row["endpoint_close_but_wrong_branch"] for row in positives)}


def _write_report(metrics: dict[str, Any], calibration: list[dict[str, Any]], per_stratum: list[dict[str, Any]]) -> None:
    lines = ["# TRACEGRAPH SBPP V3-A", "", f"Status: `{metrics['status']}`", "", "Zero-training branch-aware candidate proposal over the frozen ANZA-KIR dense source.", "", "## Calibration", "", "| tau_micro | Recall@12 | Mean branches | Median | P95 |", "|---:|---:|---:|---:|---:|"]
    for row in calibration: lines.append(f"| {row['tau_micro']:.2f} | {row['branch_recall_at_12']:.6f} | {row['mean_candidate_branches']:.3f} | {row['median_candidate_branches']:.1f} | {row['p95_candidate_branches']:.1f} |")
    dev = metrics["development"]; lines += ["", "## Development", "", f"- frozen tau_micro: `{metrics['tau_micro']}`", f"- BranchCandidateRecall@12: `{dev['branch_recall_at_12']:.6f}` (Wilson 95% `{dev['branch_recall_at_12_wilson95'][0]:.6f}..{dev['branch_recall_at_12_wilson95'][1]:.6f}`)", f"- median / p95 candidate branches: `{dev['median_candidate_branches']:.1f} / {dev['p95_candidate_branches']:.1f}`", f"- endpoint-close but wrong-branch misses: `{dev['endpoint_close_but_wrong_branch']}`", "", "## Per-stratum", "", "| Stratum | N | Recall@12 |", "|---|---:|---:|"]
    for row in per_stratum: lines.append(f"| {row['stratum']} | {row['positive_sources']} | {row['branch_recall_at_12']:.6f} |")
    lines += ["", "## Taxonomy", "", f"`{json.dumps(metrics['taxonomy'], sort_keys=True)}`", "", "## Boundary", "", "No segmentation weights, threshold, P0/P1/P2, Transformer, ANZA, path, confirm metrics, CRACKS, or expert data were opened. Confirm is hash-only."]
    (RESULT / "TRACEGRAPH_SBPP_V3_A_REPORT.md").write_text("\n".join(lines) + "\n")


def _write_calibration_stop(metrics: dict[str, Any], calibration: list[dict[str, Any]]) -> None:
    lines = ["# TRACEGRAPH SBPP V3-A", "", f"Status: `{metrics['status']}`", "", "Calibration did not reach the predeclared 0.97 coverage gate, so development was not opened.", "", "| tau_micro | Recall@12 | Mean branches | Median | P95 |", "|---:|---:|---:|---:|---:|"]
    for row in calibration: lines.append(f"| {row['tau_micro']:.2f} | {row['branch_recall_at_12']:.6f} | {row['mean_candidate_branches']:.3f} | {row['median_candidate_branches']:.1f} | {row['p95_candidate_branches']:.1f} |")
    lines += ["", "No training, development evaluation, confirm metrics, P0/P1/P2, path, CRACKS, or expert data were opened."]
    (RESULT / "TRACEGRAPH_SBPP_V3_A_REPORT.md").write_text("\n".join(lines) + "\n")


def run(*, device: str = "cuda") -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True); _json(RESULT / "protocol.json", PROTOCOL); (RESULT / "protocol_hash.txt").write_text(protocol_hash() + "\n"); _json(RESULT / "source_manifest.json", _manifest())
    generator_validation = validate_generator(); _json(RESULT / "generator_validation.json", generator_validation)
    split_manifest = {split: {"size": SPLIT_SIZES[split], "seed": SPLIT_SEEDS[split], "sha256": split_hash(split), "hash_only": split == "confirm", "inference_opened": False, "metrics_opened": False} for split in SPLIT_SIZES}
    _json(RESULT / "split_manifest.json", split_manifest)
    checkpoint_before = _sha(DENSE_CHECKPOINT); model = load_frozen_source(device)
    calibration_by_tau = _run_split(model, "calibration", TAU_CANDIDATES, device=device)
    split_manifest["calibration"].update({"inference_opened": True, "metrics_opened": True})
    calibration = [{"tau_micro": tau, **_summary(rows)} for tau, rows in calibration_by_tau.items()]; _csv(RESULT / "calibration_sweep.csv", calibration)
    eligible = [row for row in calibration if row["branch_recall_at_12"] >= 0.97]
    if eligible:
        selected = min(eligible, key=lambda row: (row["mean_candidate_branches"], -row["tau_micro"])); calibration_pass = True
    else:
        selected = max(calibration, key=lambda row: (row["branch_recall_at_12"], -row["mean_candidate_branches"], row["tau_micro"])); calibration_pass = False
    freeze = {"selection_split": "calibration", "tau_micro": selected["tau_micro"], "calibration_recall_at_12": selected["branch_recall_at_12"], "calibration_pass": calibration_pass, "config_frozen_before_development": True, "development_opened": calibration_pass, "confirm_opened": False}
    _json(RESULT / "sbpp_freeze.json", freeze)
    if not calibration_pass:
        metrics = {"status": "STOP_SBPP_CALIBRATION_COVERAGE_FAIL", "protocol_sha256": protocol_hash(), "tau_micro": selected["tau_micro"], "calibration": selected, "development": None, "taxonomy": {}, "locks": PROTOCOL["locks"]}
        _json(RESULT / "metrics.json", metrics); _csv(RESULT / "development_per_case.csv", []); _csv(RESULT / "recall_vs_k.csv", []); _csv(RESULT / "per_stratum.csv", []); _csv(RESULT / "miss_taxonomy.csv", [])
        _write_calibration_stop(metrics, calibration)
    else:
        development = _run_split(model, "development", (float(selected["tau_micro"]),), device=device)[float(selected["tau_micro"])]
        split_manifest["development"].update({"inference_opened": True, "metrics_opened": True})
        _csv(RESULT / "development_per_case.csv", development); positives = [row for row in development if row["positive"]]
        curves = [{"k": k, "branch_candidate_recall": float(np.mean([row[f"branch_recalled_at_{k}"] for row in positives]))} for k in K_VALUES]; _csv(RESULT / "recall_vs_k.csv", curves)
        per_stratum = []
        for stratum in POSITIVE_STRATA:
            local = [row for row in positives if row["stratum"] == stratum]; per_stratum.append({"stratum": stratum, "positive_sources": len(local), "branch_recall_at_12": float(np.mean([row["branch_recalled_at_12"] for row in local]))})
        _csv(RESULT / "per_stratum.csv", per_stratum)
        misses = [row for row in positives if not row["branch_recalled_at_12"]]; _csv(RESULT / "miss_taxonomy.csv", misses)
        taxonomy = {name: sum(row["miss_category"] == name for row in misses) for name in ("A3", "B3", "C3", "D3", "E3", "F3")}; dev = _summary(development)
        recall_ok = dev["branch_recall_at_12"] >= 0.95; budget_ok = dev["median_candidate_branches"] <= 8.0 and dev["p95_candidate_branches"] <= 16.0; strata_ok = all(row["branch_recall_at_12"] >= 0.90 for row in per_stratum if row["stratum"] in MAIN_SAFETY_STRATA and row["positive_sources"] >= 128)
        status = "SBPP_BRANCH_COVERAGE_PASS" if recall_ok and budget_ok and strata_ok and taxonomy["F3"] == 0 else ("STOP_SBPP_CANDIDATE_BUDGET_FAIL" if not budget_ok else "STOP_SBPP_BRANCH_COVERAGE_FAIL")
        metrics = {"status": status, "protocol_sha256": protocol_hash(), "tau_micro": selected["tau_micro"], "calibration": selected, "development": dev, "recall_curve": curves, "per_stratum": per_stratum, "taxonomy": taxonomy, "gates": {"coverage": recall_ok, "budget": budget_ok, "per_stratum": strata_ok, "generator_consistency": taxonomy["F3"] == 0}, "locks": PROTOCOL["locks"]}
        _json(RESULT / "metrics.json", metrics); _write_report(metrics, calibration, per_stratum)
    _json(RESULT / "split_manifest.json", split_manifest)
    checkpoint_after = _sha(DENSE_CHECKPOINT)
    receipt = {"checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after, "training_opened": False, "optimizer_created": False, "p0_opened": False, "transformer_opened": False, "path_opened": False, "confirm_evaluated": False, "cracks_accessed": False, "expert_accessed": False}
    _json(RESULT / "zero_training_receipt.json", receipt)
    if checkpoint_before != checkpoint_after or checkpoint_before != PROTOCOL["dense_checkpoint_sha256"]: raise PermissionError("frozen dense checkpoint drift")
    return metrics
