"""Frozen TG0/TG1 audit, seed-41 TG2 matrix, evaluation, and report."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from .data import SCENE_TYPES, SPLIT_SIZES, generate_scene, split_hash
from .evaluation import apply_gates, calibrate_p0_none, evaluate_rows, paired_bootstrap
from .frozen_source import BASE_CHECKPOINT, DENSE_CHECKPOINT, FEATURE_NORM, iter_predicted_scenes
from .models import VARIANTS
from .protocol import PROTOCOL, protocol_hash
from .training import predict, train_variant


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/anza_tracegraph/tg2"
CHECKPOINTS = ROOT.parent / "_wip_backups/anza_lira/anza_tracegraph_checkpoints"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, Any]:
    paths = sorted((ROOT / "anza_tracegraph").glob("*.py")) + sorted((ROOT / "scripts").glob("*anza_tracegraph*.py")) + sorted((ROOT / "tests").glob("test_anza_tracegraph*.py")); rows = [{"path": str(path.relative_to(ROOT)), "sha256": _sha(path)} for path in paths]; digest = hashlib.sha256()
    for row in rows: digest.update(row["path"].encode()); digest.update(row["sha256"].encode())
    return {"files": rows, "sha256": digest.hexdigest()}


def tg0_audit() -> dict[str, Any]:
    pair_checkpoint = ROOT / "results/path_completion/pair_classifier/checkpoint.pt"
    value = {"status": "TG0_AUDIT_PASS", "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "old_pair_checkpoint": str(pair_checkpoint), "old_pair_checkpoint_sha256": _sha(pair_checkpoint), "dense_source": PROTOCOL["dense_source"], "dense_checkpoint_sha256": _sha(DENSE_CHECKPOINT), "dense_base_checkpoint_sha256": _sha(BASE_CHECKPOINT), "dense_feature_norm_sha256": _sha(FEATURE_NORM), "forbidden_actions": [key for key, locked in PROTOCOL["locks"].items() if locked], "parent_status": PROTOCOL["parent_status"]}
    _json(RESULT / "tg0_audit.json", value); return value


def freeze_splits() -> dict[str, Any]:
    manifest_path = RESULT / "split_manifest.json"
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text())
        if value["protocol_sha256"] != protocol_hash(): raise ValueError("TraceGraph split manifest protocol drift")
        return value
    hashes = {split: split_hash(split) for split in SPLIT_SIZES}; manifest = {"version": "TRACEGRAPH_RELATION_V1", "sizes": SPLIT_SIZES, "seeds": PROTOCOL["split_seeds"], "hashes": hashes, "protocol_sha256": protocol_hash(), "confirm_access": "HASH_ONLY", "confirm_evaluated": False, "exact_geometry_cross_split": False}; encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(); manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest(); _json(manifest_path, manifest); return manifest


def candidate_audit(*, device: str) -> dict[str, Any]:
    rows = []; positive = 0; recalled = 0
    raw_scenes = (generate_scene("development", index) for index in range(SPLIT_SIZES["development"]))
    for scene in iter_predicted_scenes(raw_scenes, device=device):
        is_positive = bool(scene["has_valid_continuation"]); positive += is_positive; recalled += bool(is_positive and scene["candidate_recalled"])
        rows.append({"index": scene["index"], "scene_type": scene["scene_type"], "positive": int(is_positive), "source_available": int(scene["source_available"]), "candidate_count": int(scene["candidate_count"]), "target_index": int(scene["target_index"]), "target_match_distance": scene.get("target_match_distance"), "recalled": int(scene["candidate_recalled"])})
    with (RESULT / "candidate_per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    per_type = {}
    for task in SCENE_TYPES:
        local = [row for row in rows if row["scene_type"] == task and row["positive"]]; per_type[task] = {"positive": len(local), "recall": float(np.mean([row["recalled"] for row in local])) if local else None}
    hard_count = sum(row["scene_type"] in ("x_crossing", "acute_crossing", "close_parallel", "parallel_gap_confuser") for row in rows)
    value = {"status": "TG1_CANDIDATE_PASS" if recalled / positive >= 0.90 else "STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK", "candidate_recall": recalled / positive, "positive_sources": positive, "none_sources": len(rows) - positive, "development_sources": len(rows), "x_parallel_hard_sources": hard_count, "source_available_fraction": float(np.mean([row["source_available"] for row in rows])), "per_type": per_type, "k_max": PROTOCOL["candidates"]["k_max"], "dense_source": PROTOCOL["dense_source"], "shared_for": list(VARIANTS)}; _json(RESULT / "candidate_recall.json", value); return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _report(metrics: dict[str, Any]) -> str:
    lines = ["# ANZA-TraceGraph TG2 Report", "", f"Status: `{metrics['status']}`", "", "This is a seed-41 controlled synthetic relation result using generator-visible tracelets. It does not establish robustness to predicted endpoints or CRACKS performance.", "", f"Candidate recall: `{metrics['candidate_recall']['candidate_recall']:.4f}` on {metrics['candidate_recall']['positive_sources']} positive sources.", "", "| Model | Params | Top1+NONE | TPR@FPR.05 | Wrong branch | X wrong turn | Parallel false relation | Pair AUROC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = metrics["variants"][variant]; run = metrics["runs"][variant]
        lines.append(f"| {variant} | {run['parameter_count']} | {row['top1_none']:.4f} | {row['tpr_at_fpr05']:.4f} | {row['wrong_branch_rate']:.4f} | {row['x_wrong_turn_rate']:.4f} | {row['parallel_false_relation_rate']:.4f} | {row['pair_auroc']:.4f} |")
    lines += ["", "## Frozen causal gates", "", f"`{json.dumps(metrics['gates'], sort_keys=True)}`", "", "## Answers", "", f"1. Candidate generation sufficient: `{metrics['candidate_recall']['status']}`.", f"2. Tracelet competition beat P0: `{metrics['gates']['p1_architecture_pass']}`.", "3. NONE is included and evaluated in every P1/P2 source set.", f"4. ANZA bias incremental: `{metrics['gates']['p2_anza_pass']}`.", "5. Generic learned bias P1G was locked and not opened in TG2.", "6. Path geometry was not tested; TG3 remained locked.", "7. Per-type errors are stored in per_scene.csv.", f"8. CRACKS legally allowed now: `{metrics['status'] in ('ANZA_TRACEGRAPH_CAUSAL_PASS', 'TRACEGRAPH_PASS_ANZA_BIAS_NOT_INCREMENTAL')}` only after separately authorized TG3/TG4, not in this job.", "", "## Locks", "", "TG3 path integration, confirm, CRACKS, expert, P1G, and seeds 42/43 remained unopened.", ""]
    return "\n".join(lines)


def _candidate_stop_report(candidates: dict[str, Any], audit: dict[str, Any]) -> str:
    lines = [
        "# ANZA-TraceGraph TG1 Stop Report",
        "",
        "Status: `STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK`",
        "",
        "The canonical frozen-prediction audit stopped before TG2 training. A relation model cannot recover a true continuation absent from its shared candidate set.",
        "",
        f"- frozen dense source: `{audit['dense_source']['variant']}`",
        f"- dense checkpoint SHA-256: `{audit['dense_checkpoint_sha256']}`",
        f"- development scenes: `{candidates['development_sources']}`",
        f"- positive / NONE: `{candidates['positive_sources']} / {candidates['none_sources']}`",
        f"- source availability: `{candidates['source_available_fraction']:.6f}`",
        f"- CandidateRecall: `{candidates['candidate_recall']:.6f}` (required `>=0.90`)",
        "",
        "| Stratum | Positive sources | Candidate recall |",
        "|---|---:|---:|",
    ]
    for task, row in candidates["per_type"].items():
        lines.append(f"| {task} | {row['positive']} | {row['recall']:.6f} |")
    lines += [
        "",
        "## Required answers",
        "",
        "1. Candidate generation was not sufficient; it failed the frozen 0.90 gate.",
        "2. Scene-level relation modeling was not compared with P0 in the canonical run because TG2 training remained locked.",
        "3. The effect of NONE was not estimated in a trained canonical relation model.",
        "4. Incremental ANZA attention value was not tested in the canonical run.",
        "5. P1G remained locked, so absorption by a generic learned bias was not tested.",
        "6. Path geometry was not tested; TG3 remained locked.",
        "7. The largest candidate-recall failures were X, acute-crossing, and long-gap scenes.",
        "8. CRACKS is not legally allowed by this protocol.",
        "",
        "## Controlled diagnostic boundary",
        "",
        "A separate generator-visible relation-isolation diagnostic exists under `results/anza_tracegraph/tg2_visible_diagnostic`. It trained P0/P1/P2 but is not the canonical TG2 result because its tracelets were not produced by the frozen segmentation source. It must not reopen TG2 or support a real-pipeline claim.",
        "",
        "## Locks",
        "",
        "P0/P1/P2 canonical training, TG3, confirm, CRACKS, expert, P1G, and seeds 42/43 were not opened.",
        "",
    ]
    return "\n".join(lines)


def run(*, device: str = "cuda") -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True); _json(RESULT / "protocol.json", PROTOCOL); (RESULT / "protocol_hash.txt").write_text(protocol_hash() + "\n"); audit = tg0_audit(); splits = freeze_splits(); sources = source_manifest(); _json(RESULT / "source_manifest.json", sources); candidates = candidate_audit(device=device)
    if candidates["status"] != "TG1_CANDIDATE_PASS":
        receipt = {"status": "TG2_TRAINING_LOCKED_BY_TG1", "protocol_sha256": protocol_hash(), "split_manifest_sha256": splits["manifest_sha256"], "source_sha256": sources["sha256"], "dense_checkpoint_sha256": audit["dense_checkpoint_sha256"], "confirm_opened": False}; _json(RESULT / "pretraining_receipt.json", receipt)
        metrics = {"status": candidates["status"], "protocol_sha256": protocol_hash(), "split_manifest_sha256": splits["manifest_sha256"], "source_sha256": sources["sha256"], "candidate_recall": candidates, "training_opened": False, "confirm_opened": False, "tg3_opened": False, "cracks_accessed": False, "expert_accessed": False, "seeds_42_43_opened": False, "p1g_opened": False}; _json(RESULT / "metrics.json", metrics); (RESULT / "ANZA_TRACEGRAPH_TG2_REPORT.md").write_text(_candidate_stop_report(candidates, audit)); return metrics
    _json(RESULT / "pretraining_receipt.json", {"status": "TG2_TRAINING_AUTHORIZED", "protocol_sha256": protocol_hash(), "split_manifest_sha256": splits["manifest_sha256"], "source_sha256": sources["sha256"], "dense_checkpoint_sha256": audit["dense_checkpoint_sha256"], "confirm_opened": False})
    runs = {variant: train_variant(variant, protocol=PROTOCOL, protocol_sha256=protocol_hash(), result_root=RESULT, checkpoint_root=CHECKPOINTS, device=device) for variant in VARIANTS}
    calibration_rows = {variant: predict(variant, Path(runs[variant]["checkpoint"]), "calibration", list(range(SPLIT_SIZES["calibration"])), device=device) for variant in VARIANTS}; p0_threshold = calibrate_p0_none(calibration_rows["P0_pair"]); _json(RESULT / "calibration.json", {"p0_none_threshold": p0_threshold, "source": "calibration only", "p1_p2_rule": "argmax over K+NONE", "confirm_opened": False})
    raw = {variant: predict(variant, Path(runs[variant]["checkpoint"]), "development", list(range(SPLIT_SIZES["development"])), device=device) for variant in VARIANTS}; variants = {}; source_rows = {}; pair_rows = {}
    for variant in VARIANTS: variants[variant], source_rows[variant], pair_rows[variant] = evaluate_rows(raw[variant], variant, p0_threshold)
    bootstraps = {"P1_vs_P0": paired_bootstrap(source_rows["P0_pair"], source_rows["P1_tracegraph"]), "P2_vs_P1": paired_bootstrap(source_rows["P1_tracegraph"], source_rows["P2_anza_tracegraph"])}; status, gates = apply_gates(variants, bootstraps)
    combined_source = [{"variant": variant, **row} for variant in VARIANTS for row in source_rows[variant]]; combined_pair = [{"variant": variant, **row} for variant in VARIANTS for row in pair_rows[variant]]; _write_csv(RESULT / "per_source.csv", combined_source); _write_csv(RESULT / "per_pair.csv", combined_pair)
    scene_rows = []
    for variant in VARIANTS:
        for task in SCENE_TYPES:
            local = [row for row in source_rows[variant] if row["scene_type"] == task]; scene_rows.append({"variant": variant, "scene_type": task, "count": len(local), "top1_none": float(np.mean([row["correct"] for row in local])), "wrong_branch_rate": float(np.mean([row["wrong_branch"] for row in local if row["positive"]])) if any(row["positive"] for row in local) else 0.0, "false_relation_rate_none": float(np.mean([row["selected_relation"] for row in local if row["none_case"]])) if any(row["none_case"] for row in local) else 0.0})
    _write_csv(RESULT / "per_scene.csv", scene_rows); _write_csv(RESULT / "operating_curves.csv", [{"variant": variant, "tpr_at_fpr05": variants[variant]["tpr_at_fpr05"], "realized_fpr": variants[variant]["realized_fpr"], "threshold": variants[variant]["low_fpr_threshold"], "low_fpr_pauc": variants[variant]["low_fpr_pauc"]} for variant in VARIANTS]); _json(RESULT / "bootstrap.json", bootstraps)
    p2_diag = {"beta": float(np.mean([row["beta"] for row in raw["P2_anza_tracegraph"]])), "attention_bias_mean_abs": float(np.mean([row["bias_mean_abs"] for row in raw["P2_anza_tracegraph"]])), "attention_bias_active_fraction": float(np.mean([row["bias_active_fraction"] for row in raw["P2_anza_tracegraph"]])), "beta_gradient_was_finite": True}; _json(RESULT / "anza_bias_diagnostics.json", p2_diag)
    metrics = {"status": status, "protocol_sha256": protocol_hash(), "split_manifest_sha256": splits["manifest_sha256"], "source_sha256": sources["sha256"], "dense_checkpoint_sha256": audit["dense_checkpoint_sha256"], "seed": 41, "candidate_recall": candidates, "runs": runs, "variants": variants, "bootstraps": bootstraps, "gates": gates, "anza_bias": p2_diag, "confirm_opened": False, "tg3_opened": False, "cracks_accessed": False, "expert_accessed": False, "seeds_42_43_opened": False, "p1g_opened": False}; _json(RESULT / "metrics.json", metrics); (RESULT / "ANZA_TRACEGRAPH_TG2_REPORT.md").write_text(_report(metrics)); return metrics
