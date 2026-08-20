"""Zero-training SBPP V3-B soft-support calibration and gated development."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from anza_tracegraph.data_v2.generator import generate_scene
from anza_tracegraph.data_v2.strata import MAIN_SAFETY_STRATA, POSITIVE_STRATA, SPLIT_SIZES
from anza_tracegraph.frozen_source import DENSE_CHECKPOINT, infer_dense, load_frozen_source
from anza_tracegraph.ports_v3.metrics import branch_match, wilson_interval
from anza_tracegraph.ports_v3.runner import _context as hard_context, _evaluate as hard_evaluate

from .candidates import propose_cluster_candidates
from .clustering import BranchCluster, cluster_branches
from .repair_data import REPAIR_CALIBRATION_SEED, REPAIR_CALIBRATION_SIZE, generate_repair_scene, repair_calibration_hash
from .soft_branches import SoftBranch, extract_soft_branches


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/anza_tracegraph/sbpp_v3_b"
PARENT = ROOT / "results/anza_tracegraph/sbpp_v3_a"
TAU_VALUES = (0.30, 0.25, 0.20)
K_VALUES = (4, 8, 12, 16)
PROTOCOL: dict[str, Any] = {
    "version": "TRACEGRAPH_SBPP_V3_B_SOFT_SUPPORT",
    "parent_status": "STOP_SBPP_CALIBRATION_COVERAGE_FAIL",
    "dense_checkpoint_sha256": "95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6",
    "hard_threshold": 0.35,
    "soft_thresholds": list(TAU_VALUES),
    "selection": "highest tau_s satisfying every repair-calibration gate",
    "sector": {"distance_px": [6.0, 68.0], "directed": True, "maximum_angle_degrees": 78.0},
    "hysteresis": {"H1_hard_distance_px": 3.0, "H2_min_length_px": 6.0, "H2_probability_margin": 0.03, "H2_axis_coherence": 0.60},
    "clustering": {"overlap_radius_px": 2.0, "overlap_fraction": 0.60, "maximum_median_axial_mismatch_degrees": 30.0, "truth_free": True},
    "candidate": {"primary_k": 12, "curve_k": list(K_VALUES), "dedup": "one per soft/hard cluster"},
    "calibration_gates": {"branch_recall_at_12": 0.970, "median_candidates": 4.0, "p95_candidates": 12.0, "wrong_endpoint_rate_ratio": 1.25, "B6": 0},
    "development_gates": {"branch_recall_at_12": 0.950, "median_candidates": 8.0, "p95_candidates": 16.0, "main_stratum_recall": 0.90, "B6": 0},
    "locks": {"training": True, "p0": True, "transformer": True, "anza": True, "path": True, "confirm": True, "cracks": True, "expert": True},
}


def _json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fieldnames = list(rows[0]) + sorted({key for row in rows for key in row}.difference(rows[0]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def protocol_hash() -> str: return hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_manifest() -> dict[str, Any]:
    paths = [ROOT / path for path in ("anza_tracegraph/ports_v3_b/repair_data.py", "anza_tracegraph/ports_v3_b/soft_branches.py", "anza_tracegraph/ports_v3_b/clustering.py", "anza_tracegraph/ports_v3_b/candidates.py", "anza_tracegraph/ports_v3_b/runner.py", "anza_tracegraph/ports_v3_b/validator.py", "scripts/run_tracegraph_sbpp_v3_b.py", "scripts/validate_tracegraph_sbpp_v3_b.py", "tests/test_tracegraph_sbpp_v3_b_soft.py", "tests/test_tracegraph_sbpp_v3_b_protocol.py")]
    rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]; digest = hashlib.sha256()
    for row in rows: digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": rows, "sha256": digest.hexdigest()}


def _cluster_match(cluster: BranchCluster, target: np.ndarray) -> tuple[bool, bool, bool]:
    hard = False; soft = False
    for member in cluster.members:
        matched, _, _ = branch_match(member, target)
        if matched and isinstance(member, SoftBranch): soft = True
        elif matched: hard = True
    return hard or soft, hard, soft


def _support_diagnostics(probability: np.ndarray, target: np.ndarray | None) -> dict[str, Any]:
    if target is None: return {"max_target_probability": "", "target_support_fraction_030": "", "target_support_fraction_025": "", "target_support_fraction_020": ""}
    pixels = np.rint(target).astype(int); pixels[:, 0] = np.clip(pixels[:, 0], 0, probability.shape[0] - 1); pixels[:, 1] = np.clip(pixels[:, 1], 0, probability.shape[1] - 1); values = probability[pixels[:, 0], pixels[:, 1]]
    return {"max_target_probability": float(values.max()), "target_support_fraction_030": float(np.mean(values >= 0.30)), "target_support_fraction_025": float(np.mean(values >= 0.25)), "target_support_fraction_020": float(np.mean(values >= 0.20))}


def evaluate_soft(scene: dict[str, Any], probability: np.ndarray, tau_s: float) -> dict[str, Any]:
    hard = hard_context(scene, probability, 0.35); truth = scene["truth"]; positive = bool(truth["has_valid_continuation"]); target = truth["destination_branch"]
    diagnostics = _support_diagnostics(probability, target)
    if hard["source"] is None:
        return {"split": scene["input"]["split"], "index": scene["input"]["index"], "stratum": scene["input"]["stratum"], "tau_s": tau_s, "positive": int(positive), "source_available": 0, "hard_branch_count": len(hard["branches"]), "soft_branch_count": 0, "cluster_count": len(hard["branches"]), "candidate_branch_count": 0, **{f"branch_recalled_at_{k}": int(not positive) for k in K_VALUES}, "correct_hard_branch_extracted": 0, "correct_soft_branch_extracted": 0, "correct_branch_valid_landing": 0, "correct_branch_best_rank": -1, "endpoint_close_but_wrong_branch": 0, "miss_category": "B0" if positive else "NONE", **diagnostics}
    excluded = np.zeros_like(probability, dtype=bool); start, end = scene["input"]["relation_corridor_x"]; excluded[:, start:end] = True
    soft = extract_soft_branches(probability, scene["input"]["model_input"][0], hard["mask"], hard["source"], tau_s=tau_s, excluded_mask=excluded)
    clusters = cluster_branches(hard["branches"], soft); candidates = propose_cluster_candidates(hard["source"], clusters)
    matches: dict[int, tuple[bool, bool, bool]] = {}
    if positive and target is not None: matches = {cluster.cluster_id: _cluster_match(cluster, target) for cluster in clusters}
    row: dict[str, Any] = {"split": scene["input"]["split"], "index": scene["input"]["index"], "stratum": scene["input"]["stratum"], "tau_s": tau_s, "positive": int(positive), "source_available": 1, "hard_branch_count": len(hard["branches"]), "soft_branch_count": len(soft), "cluster_count": len(clusters), "candidate_branch_count": len(candidates), **diagnostics}
    for k in K_VALUES: row[f"branch_recalled_at_{k}"] = int(not positive or any(matches.get(candidate.destination_branch_id, (False, False, False))[0] for candidate in candidates[:k]))
    row["correct_hard_branch_extracted"] = int(any(value[1] for value in matches.values())) if positive else 0
    row["correct_soft_branch_extracted"] = int(any(value[2] for value in matches.values())) if positive else 0
    row["correct_branch_valid_landing"] = int(any(matches.get(candidate.destination_branch_id, (False, False, False))[0] for candidate in candidates)) if positive else 0
    row["correct_branch_best_rank"] = next((rank for rank, candidate in enumerate(candidates) if matches.get(candidate.destination_branch_id, (False, False, False))[0]), -1)
    row["endpoint_close_but_wrong_branch"] = 0
    if positive and target is not None:
        endpoint = np.asarray(target[0]); close_wrong = [candidate for candidate in candidates[:12] if np.linalg.norm(np.asarray(candidate.landing_point_yx) - endpoint) <= 6.0 and not matches.get(candidate.destination_branch_id, (False, False, False))[0]]
        row["endpoint_close_but_wrong_branch"] = int(bool(close_wrong) and not row["branch_recalled_at_12"])
    if positive and not row["branch_recalled_at_12"]:
        if target is None: category = "B6"
        elif row["correct_branch_valid_landing"] and row["correct_branch_best_rank"] >= 12: category = "B3"
        elif row["correct_hard_branch_extracted"]: category = "B1"
        elif row["correct_soft_branch_extracted"]: category = "B2"
        elif float(row["max_target_probability"]) >= 0.20: category = "B4"
        else: category = "B5"
        row["miss_category"] = category
    else: row["miss_category"] = "NONE"
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["positive"]]; success = sum(row["branch_recalled_at_12"] for row in positives); counts = np.asarray([row["candidate_branch_count"] for row in rows], dtype=float); interval = wilson_interval(success, len(positives)); wrong = sum(row["endpoint_close_but_wrong_branch"] for row in positives)
    return {"sources": len(rows), "positive_sources": len(positives), "none_sources": len(rows) - len(positives), "successes_at_12": success, "branch_recall_at_12": success / len(positives), "branch_recall_wilson95": list(interval), "median_candidate_branches": float(np.median(counts)), "p95_candidate_branches": float(np.quantile(counts, 0.95)), "mean_candidate_branches": float(counts.mean()), "endpoint_close_but_wrong_branch": wrong, "endpoint_close_but_wrong_branch_rate": wrong / len(positives), "B6": sum(row["miss_category"] == "B6" for row in positives)}


def _per_stratum(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    output = []
    for stratum in POSITIVE_STRATA:
        local = [row for row in rows if row["positive"] and row["stratum"] == stratum]; success = sum(row["branch_recalled_at_12"] for row in local); interval = wilson_interval(success, len(local))
        output.append({"variant": variant, "stratum": stratum, "positive_sources": len(local), "successes": success, "branch_recall_at_12": success / len(local), "wilson95_low": interval[0], "wilson95_high": interval[1]})
    return output


def _run_stream(model: Any, generator: Any, size: int, taus: tuple[float, ...], *, device: str, batch_size: int = 64, include_hard: bool = False) -> tuple[dict[float, list[dict[str, Any]]], list[dict[str, Any]]]:
    soft_rows = {tau: [] for tau in taus}; hard_rows: list[dict[str, Any]] = []
    for start in range(0, size, batch_size):
        scenes = [generator(index) for index in range(start, min(start + batch_size, size))]; probabilities, _ = infer_dense(model, np.stack([scene["input"]["model_input"] for scene in scenes]), device=device)
        for scene, probability in zip(scenes, probabilities):
            if include_hard: hard_rows.append(hard_evaluate(scene, probability, 0.35))
            for tau in taus: soft_rows[tau].append(evaluate_soft(scene, probability, tau))
    return soft_rows, hard_rows


def _artifact_rows(rows_by_tau: dict[float, list[dict[str, Any]]], hard_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_case = [row for tau in TAU_VALUES for row in rows_by_tau[tau]]; per_stratum = [row for tau in TAU_VALUES for row in _per_stratum(rows_by_tau[tau], f"soft_{tau:.2f}")]
    taxonomy = [row for row in per_case if row["positive"] and not row["branch_recalled_at_12"]]
    recall = []
    hard_positive = [row for row in hard_rows if row["positive"]]
    for k in K_VALUES: recall.append({"variant": "hard_reference", "k": k, "branch_candidate_recall": float(np.mean([row[f"branch_recalled_at_{k}"] for row in hard_positive]))})
    for tau in TAU_VALUES:
        positives = [row for row in rows_by_tau[tau] if row["positive"]]
        for k in K_VALUES: recall.append({"variant": f"soft_{tau:.2f}", "k": k, "branch_candidate_recall": float(np.mean([row[f"branch_recalled_at_{k}"] for row in positives]))})
    return per_case, per_stratum, taxonomy, recall


def _write_report(metrics: dict[str, Any], sweep: list[dict[str, Any]]) -> None:
    lines = ["# TRACEGRAPH SBPP V3-B", "", f"Status: `{metrics['status']}`", "", "Candidate-only soft support was evaluated without training or changing the frozen hard graph.", "", "| Variant | Recall@12 | Median | P95 | Wrong-near rate | Eligible |", "|---|---:|---:|---:|---:|---:|"]
    for row in sweep: lines.append(f"| {row['variant']} | {row['branch_recall_at_12']:.6f} | {row['median_candidate_branches']:.1f} | {row['p95_candidate_branches']:.1f} | {row['endpoint_close_but_wrong_branch_rate']:.6f} | {row.get('eligible', '')} |")
    lines += ["", f"Selected tau_s: `{metrics.get('selected_tau_s')}`", ""]
    if metrics.get("development"):
        dev = metrics["development"]; lines += ["## Development", "", f"- successes: `{dev['successes_at_12']}/{dev['positive_sources']}`", f"- BranchCandidateRecall@12: `{dev['branch_recall_at_12']:.6f}`", f"- Wilson 95%: `{dev['branch_recall_wilson95'][0]:.6f}..{dev['branch_recall_wilson95'][1]:.6f}`", f"- median / p95 candidates: `{dev['median_candidate_branches']:.1f} / {dev['p95_candidate_branches']:.1f}`", f"- miss taxonomy: `{json.dumps(dev['taxonomy'], sort_keys=True)}`", "", "| Stratum | N | Success | Recall@12 | Wilson 95% |", "|---|---:|---:|---:|---:|"]
        for row in metrics["development_per_stratum"]: lines.append(f"| {row['stratum']} | {row['positive_sources']} | {row['successes']} | {row['branch_recall_at_12']:.6f} | {row['wilson95_low']:.6f}..{row['wilson95_high']:.6f} |")
        lines += ["", "`weak_branch_continue` remains a localized failure (not one of the predeclared V3-B main-stratum gates); no weak-branch success claim is permitted.", ""]
    else: lines += ["Development remained unopened because repair calibration did not pass every frozen gate.", ""]
    lines += ["## Boundary", "", "No P0/P1/P2, Transformer, ANZA, path, confirm metrics, CRACKS, expert data, optimizer, or training was opened."]
    (RESULT / "TRACEGRAPH_SBPP_V3_B_REPORT.md").write_text("\n".join(lines) + "\n")


def run(*, device: str = "cuda") -> dict[str, Any]:
    if (RESULT / "development_per_case.csv").exists(): raise PermissionError("V3-B development was already opened once")
    RESULT.mkdir(parents=True, exist_ok=True); _json(RESULT / "protocol.json", PROTOCOL); (RESULT / "protocol_hash.txt").write_text(protocol_hash() + "\n"); _json(RESULT / "source_manifest.json", _source_manifest())
    parent_metrics = json.loads((PARENT / "metrics.json").read_text()); parent_split = json.loads((PARENT / "split_manifest.json").read_text())
    repair_hash = repair_calibration_hash(); split_manifest = {"repair_calibration": {"seed": REPAIR_CALIBRATION_SEED, "size": REPAIR_CALIBRATION_SIZE, "sha256": repair_hash, "hash_frozen_before_evaluation": True, "inference_opened": False}, "development": {**parent_split["development"], "inference_opened": False, "metrics_opened": False}, "confirm": {**parent_split["confirm"], "inference_opened": False, "metrics_opened": False}}
    _json(RESULT / "split_manifest.json", split_manifest); _json(RESULT / "old_calibration_forensic.json", {"source": "immutable V3-A metrics; no V3-B selection", "status": parent_metrics["status"], "positive_sources": parent_metrics["calibration"]["positive_sources"], "branch_recall_at_12": parent_metrics["calibration"]["branch_recall_at_12"], "development_opened": False})
    checkpoint_before = _sha(DENSE_CHECKPOINT); model = load_frozen_source(device)
    soft_rows, hard_rows = _run_stream(model, generate_repair_scene, REPAIR_CALIBRATION_SIZE, TAU_VALUES, device=device, include_hard=True); split_manifest["repair_calibration"]["inference_opened"] = True
    hard_summary = _summary(hard_rows); hard_summary.update({"variant": "hard_reference", "eligible": "reference"}); sweep = [hard_summary]
    for tau in TAU_VALUES:
        summary = _summary(soft_rows[tau]); wrong_safe = summary["endpoint_close_but_wrong_branch_rate"] <= 1.25 * hard_summary["endpoint_close_but_wrong_branch_rate"] if hard_summary["endpoint_close_but_wrong_branch_rate"] > 0 else summary["endpoint_close_but_wrong_branch_rate"] == 0
        eligible = summary["branch_recall_at_12"] >= 0.970 and summary["median_candidate_branches"] <= 4.0 and summary["p95_candidate_branches"] <= 12.0 and wrong_safe and summary["B6"] == 0
        summary.update({"variant": f"soft_{tau:.2f}", "tau_s": tau, "wrong_endpoint_safe": wrong_safe, "eligible": eligible}); sweep.append(summary)
    per_case, per_stratum, taxonomy, recall = _artifact_rows(soft_rows, hard_rows); _csv(RESULT / "repair_calibration_sweep.csv", sweep); _csv(RESULT / "repair_calibration_per_case.csv", per_case); _csv(RESULT / "repair_calibration_per_stratum.csv", per_stratum); _csv(RESULT / "repair_calibration_taxonomy.csv", taxonomy); _csv(RESULT / "repair_calibration_recall_vs_k.csv", recall); _csv(RESULT / "repair_calibration_candidate_burden.csv", [{key: row[key] for key in ("variant", "median_candidate_branches", "p95_candidate_branches", "mean_candidate_branches")} for row in sweep])
    selected_rows = [row for row in sweep[1:] if row["eligible"]]; selected = selected_rows[0] if selected_rows else None
    freeze = {"selection_split": "repair_calibration", "repair_calibration_sha256": repair_hash, "selected_tau_s": None if selected is None else selected["tau_s"], "calibration_pass": selected is not None, "highest_eligible_rule": True, "development_opened": False, "confirm_opened": False}; _json(RESULT / "sbpp_v3_b_freeze.json", freeze)
    development_summary = None; development_per_stratum: list[dict[str, Any]] = []; development_taxonomy: list[dict[str, Any]] = []
    if selected is None:
        status = "STOP_SBPP_V3_B_SOFT_SUPPORT_FAIL"
    else:
        dev_rows_by_tau, _ = _run_stream(model, lambda index: generate_scene("development", index), SPLIT_SIZES["development"], (float(selected["tau_s"]),), device=device); dev_rows = dev_rows_by_tau[float(selected["tau_s"])]
        _csv(RESULT / "development_per_case.csv", dev_rows); development_per_stratum = _per_stratum(dev_rows, f"soft_{selected['tau_s']:.2f}"); _csv(RESULT / "development_per_stratum.csv", development_per_stratum); development_taxonomy = [row for row in dev_rows if row["positive"] and not row["branch_recalled_at_12"]]; _csv(RESULT / "development_taxonomy.csv", development_taxonomy)
        dev_positive = [row for row in dev_rows if row["positive"]]; dev_curve = [{"k": k, "branch_candidate_recall": float(np.mean([row[f"branch_recalled_at_{k}"] for row in dev_positive]))} for k in K_VALUES]; _csv(RESULT / "recall_vs_k.csv", dev_curve)
        development_summary = _summary(dev_rows); taxonomy_counts = {name: sum(row["miss_category"] == name for row in development_taxonomy) for name in ("B0", "B1", "B2", "B3", "B4", "B5", "B6")}
        strata_safe = all(row["branch_recall_at_12"] >= 0.90 for row in development_per_stratum if row["stratum"] in MAIN_SAFETY_STRATA and row["positive_sources"] >= 128)
        passed = development_summary["branch_recall_at_12"] >= 0.95 and development_summary["median_candidate_branches"] <= 8 and development_summary["p95_candidate_branches"] <= 16 and strata_safe and taxonomy_counts["B6"] == 0
        status = "SBPP_V3_B_BRANCH_COVERAGE_PASS" if passed else "STOP_SBPP_V3_B_DEVELOPMENT_FAIL"; development_summary.update({"taxonomy": taxonomy_counts, "main_strata_safe": strata_safe}); split_manifest["development"].update({"inference_opened": True, "metrics_opened": True}); freeze["development_opened"] = True; _json(RESULT / "sbpp_v3_b_freeze.json", freeze)
    metrics = {"status": status, "protocol_sha256": protocol_hash(), "hard_reference": hard_summary, "repair_calibration": sweep[1:], "selected_tau_s": None if selected is None else selected["tau_s"], "development": development_summary, "development_per_stratum": development_per_stratum, "locks": PROTOCOL["locks"]}; _json(RESULT / "metrics.json", metrics); _json(RESULT / "split_manifest.json", split_manifest)
    checkpoint_after = _sha(DENSE_CHECKPOINT); receipt = {"checkpoint_before": checkpoint_before, "checkpoint_after": checkpoint_after, "training_opened": False, "optimizer_created": False, "p0_opened": False, "transformer_opened": False, "anza_opened": False, "path_opened": False, "confirm_evaluated": False, "cracks_accessed": False, "expert_accessed": False}; _json(RESULT / "zero_training_receipt.json", receipt)
    if checkpoint_before != checkpoint_after or checkpoint_before != PROTOCOL["dense_checkpoint_sha256"]: raise PermissionError("frozen dense checkpoint drift")
    _write_report(metrics, sweep); return metrics
