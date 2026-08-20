"""Bounded Phase 3D-A/B audit and zero-training oracle runner."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from anza2.eval.low_fpr import sampled_operating_curve
from synthetic.crossing_trace_bench_v4 import SPLIT_SIZES_V4, benchmark_v4_config

from .case_manifest import build_complete_manifest, split_summary, write_manifest
from .context_sufficiency import context_sufficiency
from .oracle_graph_eval import METHODS, calibrate_thresholds, evaluate_oracle_rows, oracle_rows
from .structural_sampler import MANDATORY_STRATA, balanced_curriculum_indices, strata_inventory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3d_ab"
PARENT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase3c_b_rc1"
PHASE_VERSION = "ANZA2_PHASE3D_CONTEXT_MODE_STATE_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _git_state() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = call("git", "status", "--short")
    return {
        "branch": call("git", "branch", "--show-current"),
        "head": call("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "commit_created": False,
        "push_performed": False,
    }


def protocol_payload() -> dict[str, Any]:
    parent_validator = json.loads((PARENT_ROOT / "validator.json").read_text())
    if parent_validator.get("research_status") != "STOP_RC1_MEMBERSHIP_REPAIR_FAILED":
        raise ValueError("frozen RC1 parent status changed")
    return {
        "version": PHASE_VERSION,
        "parent_phase": "phase3c_b_rc1",
        "parent_protocol_sha256": file_hash(PARENT_ROOT / "protocol.json"),
        "parent_validator_sha256": file_hash(PARENT_ROOT / "validator.json"),
        "parent_research_status": parent_validator["research_status"],
        "benchmark": benchmark_v4_config(),
        "manifest_scope": {split: int(size) for split, size in SPLIT_SIZES_V4.items() if split != "test"},
        "mandatory_strata": MANDATORY_STRATA,
        "stratum_name_resolution": "Master shorthand CurvedGap is represented by frozen-v4 curved_fault; no new case was fabricated.",
        "target_contract": {
            "local_mode_supervision": "visible_fault_mask only",
            "latent_positive_gap": "continuation/reachability target only",
            "privileged_latent_local_supervision": False,
        },
        "graph_contract": {
            "node": "(pixel, mode)",
            "edge": "sqrt(mu_r(p)*mu_s(q)*G_r(p->q)*G_s(q->p))",
            "free_intra_pixel_mode_switch": False,
            "algorithm": "max-min widest path",
        },
        "oracle_comparison": {
            "methods": list(METHODS),
            "calibration": "all eligible train[0:512] rows",
            "evaluation": "all eligible validation[0:512] rows",
            "same_field_domain_algorithm_threshold_rule": True,
            "maximum_fpr": 0.05,
        },
        "gate": {
            "positive_recall_noninferiority_margin": 0.01,
            "x_wrong_turn_relative_reduction_minimum": 0.50,
            "parallel_false_bridge_noninferiority": True,
            "negative_gap_false_bridge_noninferiority": True,
            "curved_recall_noninferiority_margin": 0.01,
        },
        "training_performed": False,
        "confirm_manifest_access_only": True,
        "confirm_evaluation_opened": False,
        "cracks_data_accessed": False,
        "expert_data_accessed": False,
    }


def _target_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"splits": {}}
    for split in ("train", "validation", "confirm"):
        selected = [row for row in rows if row["split"] == split]
        result["splits"][split] = {
            key: int(sum(int(row[key]) for row in selected))
            for key in (
                "original_target_axes", "visible_target_axes", "latent_gap_target_axes",
                "unobserved_non_gap_target_axes", "visible_latent_overlap_axes",
            )
        }
        result["splits"][split]["all_privileged_gap_local_supervision_removed"] = all(
            bool(row["privileged_gap_local_supervision_removed"]) for row in selected
        )
    result["pass"] = all(
        values["visible_latent_overlap_axes"] == 0 and values["all_privileged_gap_local_supervision_removed"]
        for values in result["splits"].values()
    )
    result["claim_boundary"] = (
        "Oracle fields may use latent axes only for mathematical feasibility; learned local membership supervision may not."
    )
    return result


def _case_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        threshold = thresholds[method]
        for task in sorted({row["task"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row["task"] == task]
            output.append({
                "method": method,
                "task": task,
                "label": int(selected[0]["label"]),
                "count": len(selected),
                "mean_score": float(np.mean([row["score"] for row in selected])),
                "acceptance_rate": float(np.mean([row["score"] >= threshold for row in selected])),
                "threshold_from_train": threshold,
            })
    return output


def _operating_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        positive = np.asarray([row["score"] for row in selected if row["label"] == 1])
        negative = np.asarray([row["score"] for row in selected if row["label"] == 0])
        for point in sampled_operating_curve(positive, negative):
            output.append({"split": "validation", "method": method, **point})
    return output


def _report(metrics: dict[str, Any], context: dict[str, Any], manifest: dict[str, Any]) -> str:
    scalar = metrics["methods"]["G0_scalar"]
    state = metrics["methods"]["G1_mode_state"]
    status = metrics["status"]
    next_action = (
        "Phase 3D-C is authorized by the frozen master protocol, but was not run in this phase."
        if metrics["gate_pass"] else
        "Final scientific STOP for mode-state reachability; Phase 3D-C, confirm, CRACKS, and expert remain forbidden."
    )
    return "\n".join([
        "# ANZA-2 Phase 3D-A/B report", "", "## Status", "", f"`{status}`", "",
        "No training was performed. The complete frozen v4 train/validation/confirm composition was inventoried; confirm labels were used only for metadata counts and never for scoring or threshold selection.", "",
        "## Data and target audit", "",
        f"- Manifest: {sum(item['count'] for item in manifest['splits'].values())} samples, exactly 512 per split.",
        f"- Pairwise seed overlap: `{manifest['seed_overlap']}`.",
        "- Local mode supervision is restricted to visible evidence. Positive-gap latent axes are reserved for continuation/oracle feasibility.",
        f"- Frozen encoder receptive field: {context['encoder_effective_receptive_field_px']} px; required q90 local scale: {context['required_context_q90_px']:.3f} px; action: `{context['architecture_action']}`.", "",
        "## Frozen validation comparison", "",
        "| Metric | G0 scalar | G1 mode-state |", "|---|---:|---:|",
        f"| Positive continuation recall | {scalar['validation_positive_continuation_recall']:.4f} | {state['validation_positive_continuation_recall']:.4f} |",
        f"| Overall negative FPR | {scalar['validation_overall_negative_fpr']:.4f} | {state['validation_overall_negative_fpr']:.4f} |",
        f"| X correct recall | {scalar['x_correct_recall']:.4f} | {state['x_correct_recall']:.4f} |",
        f"| X wrong-turn FPR | {scalar['x_wrong_turn_fpr']:.4f} | {state['x_wrong_turn_fpr']:.4f} |",
        f"| Curved continuation recall | {scalar['curved_continuation_recall']:.4f} | {state['curved_continuation_recall']:.4f} |",
        f"| Parallel false bridge | {scalar['parallel_false_bridge']:.4f} | {state['parallel_false_bridge']:.4f} |",
        f"| Positive-gap recovery | {scalar['positive_gap_recovery']:.4f} | {state['positive_gap_recovery']:.4f} |",
        f"| Negative-gap false bridge | {scalar['negative_gap_false_bridge']:.4f} | {state['negative_gap_false_bridge']:.4f} |", "",
        f"X wrong-turn relative reduction: `{metrics['x_wrong_turn_relative_reduction']:.4f}` (required >= 0.50).", "",
        f"Gate checks: `{metrics['gate_checks']}`.", "",
        "## Decision", "", next_action, "",
        "This oracle result is a controlled mathematical feasibility test, not a trained-model or CRACKS performance claim. No result was tuned to become positive.", "",
    ])


def run(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(); encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists() and protocol_path.read_text() != encoded:
        raise ValueError("Phase 3D-A/B protocol drift")
    protocol_path.write_text(encoded)
    protocol_sha = canonical_hash(protocol)
    (output_root / "protocol_hash.txt").write_text(protocol_sha + "\n")
    _write_json(output_root / "parent_evidence.json", {
        "phase": "phase3c_b_rc1",
        "protocol_sha256": file_hash(PARENT_ROOT / "protocol.json"),
        "validator_sha256": file_hash(PARENT_ROOT / "validator.json"),
        "validator": json.loads((PARENT_ROOT / "validator.json").read_text()),
    })
    _write_json(output_root / "code_state.json", _git_state())
    _write_json(output_root / "data_access_log.json", {
        "synthetic_train_manifest": True, "synthetic_validation_manifest": True,
        "synthetic_confirm_manifest_metadata_only": True,
        "synthetic_train_scored_for_thresholds": True,
        "synthetic_validation_scored_for_gate": True,
        "synthetic_confirm_scored": False, "cracks": False, "expert": False,
    })

    print("phase=3D-A step=manifest samples=1536 training=NO confirm_metrics=CLOSED", flush=True)
    manifest_rows = build_complete_manifest(image_size=64)
    write_manifest(output_root / "case_manifest.csv", manifest_rows)
    write_manifest(output_root / "PHASE3D_CASE_MANIFEST.csv", manifest_rows)
    manifest = split_summary(manifest_rows)
    inventories = {split: strata_inventory(manifest_rows, split=split) for split in ("train", "validation", "confirm")}
    manifest["mandatory_strata_counts"] = {
        split: {name: len(values) for name, values in inventory.items()}
        for split, inventory in inventories.items()
    }
    manifest["all_mandatory_strata_present"] = all(
        all(count > 0 for count in counts.values()) for counts in manifest["mandatory_strata_counts"].values()
    )
    _write_json(output_root / "split_manifest.json", manifest)
    target_audit = _target_audit(manifest_rows)
    _write_json(output_root / "visible_latent_target_audit.json", target_audit)
    context = context_sufficiency(manifest_rows, split="train")
    _write_json(output_root / "CONTEXT_SUFFICIENCY.json", context)
    curriculum = balanced_curriculum_indices(manifest_rows, split="train", quota=64)
    _write_json(output_root / "strata_curriculum_manifest.json", {
        "status": "FROZEN_NOT_RUN", "quota_per_stratum": 64, "total": len(curriculum), "schedule": curriculum,
    })
    phase_a_pass = bool(
        all(manifest["splits"][split]["count"] == 512 for split in ("train", "validation", "confirm"))
        and all(value == 0 for value in manifest["seed_overlap"].values())
        and manifest["all_mandatory_strata_present"] and target_audit["pass"]
    )
    if not phase_a_pass:
        raise RuntimeError("Phase 3D-A audit failed; oracle is not authorized")

    print("phase=3D-B step=train-threshold-oracle samples=512 training=NO", flush=True)
    train_rows = oracle_rows("train", image_size=64)
    thresholds = calibrate_thresholds(train_rows)
    _write_json(output_root / "threshold_freeze.json", {
        "source": "all eligible train[0:512] oracle pairs", "maximum_fpr": 0.05,
        "thresholds": thresholds, "validation_used": False, "confirm_used": False,
    })
    print("phase=3D-B step=validation-oracle samples=512 training=NO confirm_metrics=CLOSED", flush=True)
    validation_rows = oracle_rows("validation", image_size=64)
    metrics = evaluate_oracle_rows(validation_rows, thresholds)
    metrics.update({"phase3d_a_pass": phase_a_pass, "protocol_sha256": protocol_sha})
    _write_json(output_root / "metrics.json", metrics)
    _write_csv(output_root / "mode_state_paths.csv", validation_rows)
    _write_csv(output_root / "per_candidate.csv", validation_rows)
    _write_csv(output_root / "per_gap.csv", [row for row in validation_rows if "gap" in row["task"]])
    _write_csv(output_root / "per_case.csv", _case_metrics(validation_rows, thresholds))
    _write_csv(output_root / "operating_curve.csv", _operating_rows(validation_rows))
    _write_json(output_root / "bootstrap.json", {
        "status": "NOT_PRIMARY_INFERENCE_ORACLE",
        "reason": "This phase is a deterministic mathematical oracle feasibility gate; no trained-model claim is made.",
        "unit": "synthetic sample", "confirm_used": False,
    })
    task_state = {
        "status": metrics["status"],
        "training_performed": False,
        "phase3d_c_authorized": bool(metrics["gate_pass"]),
        "next_action": (
            "Freeze Phase 3D-C curriculum/training protocol before any training"
            if metrics["gate_pass"] else "STOP ANZA-2 mode-state development; do not run Phase 3D-C"
        ),
        "confirm_evaluation_opened": False, "cracks_data_accessed": False, "expert_data_accessed": False,
    }
    _write_json(output_root / "TASK_STATE.json", task_state)
    _write_json(output_root / "EVIDENCE.json", {
        "status": metrics["status"], "protocol_sha256": protocol_sha,
        "manifest_samples": len(manifest_rows), "phase3d_a_pass": phase_a_pass,
        "oracle_metrics": metrics, "training_performed": False,
        "claim_boundary": "Oracle feasibility only; no learned or real-data improvement claim.",
    })
    report = _report(metrics, context, manifest)
    (output_root / "REPORT.md").write_text(report)
    (output_root / "PHASE3D_AB_REPORT.md").write_text(report)
    return metrics

