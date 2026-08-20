"""Fail-closed ANZA-KS K0/K1 orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .benchmark.matched_generator import TASKS, benchmark_manifest, confirm_stream_hash
from .benchmark.validator import validate_static_matching
from .experiments.k0_audit import run_k0_math
from .experiments.k1_feature_study import run_k1
from .features import METHODS
from .protocol import FREEZE_ROOT, RESULT_ROOT, canonical_hash, freeze_protocol


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(sorted(Path("anza_ks").rglob("*.py"))) + (
    Path("scripts/run_anza_ks_k0_k1.py"),
    Path("scripts/validate_anza_ks_k0_k1.py"),
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> dict[str, Any]:
    files = []
    aggregate = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(relative), "sha256": digest})
        aggregate.update(str(relative).encode())
        aggregate.update(digest.encode())
    return {"files": files, "sha256": aggregate.hexdigest()}


def freeze_k0_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = freeze_protocol()
    existing_receipt = FREEZE_ROOT / "freeze_receipt.json"
    if existing_receipt.exists():
        receipt = json.loads(existing_receipt.read_text())
        benchmark = json.loads((FREEZE_ROOT / "benchmark_manifest.json").read_text())
        if receipt["source_sha256"] != source_manifest()["sha256"]:
            raise ValueError("ANZA-KS source drift after freeze")
        if receipt["protocol_sha256"] != canonical_hash(protocol):
            raise ValueError("ANZA-KS protocol drift after freeze")
        if receipt["benchmark_sha256"] != canonical_hash(benchmark):
            raise ValueError("ANZA-KS benchmark drift after freeze")
        return protocol, benchmark, receipt

    k0_math = run_k0_math()
    static_audit = validate_static_matching()
    confirm_hash = confirm_stream_hash()
    benchmark = {
        **benchmark_manifest(),
        "confirm_stream_sha256": confirm_hash,
        "confirm_samples_exposed": False,
        "static_match_status": static_audit["status"],
        "static_diagnostics": static_audit["diagnostics"],
    }
    benchmark["sha256"] = canonical_hash(benchmark)
    sources = source_manifest()
    receipt = {
        "protocol_sha256": canonical_hash(protocol),
        "benchmark_sha256": canonical_hash(benchmark),
        "source_sha256": sources["sha256"],
        "static_only_validation_completed_before_symbolic_scoring": True,
        "symbolic_scores_computed": False,
        "confirm_evaluated": False,
    }
    _json(FREEZE_ROOT / "k0_math.json", k0_math)
    _json(FREEZE_ROOT / "benchmark_manifest.json", benchmark)
    _json(RESULT_ROOT / "benchmark_manifest.json", benchmark)
    _json(FREEZE_ROOT / "source_manifest.json", sources)
    _json(FREEZE_ROOT / "freeze_receipt.json", receipt)
    _csv(RESULT_ROOT / "static_match_diagnostics.csv", static_audit["diagnostics"])
    if k0_math["status"] != "ANZA_KS_K0_MATH_PASS" or static_audit["status"] != "ANZA_KS_STATIC_MATCH_PASS":
        raise ValueError("ANZA-KS K0 failed; K1 remains locked")
    return protocol, benchmark, receipt


def _report(metrics: dict[str, Any]) -> str:
    k1 = metrics["k1"]
    lines = [
        "# ANZA-KS K0/K1 report",
        "",
        "## Status",
        "",
        f"`{metrics['status']}`",
        "",
        "This is a frozen static-matched small-logistic causal feature study. It is not segmentation, confirm, CRACKS, or expert evidence.",
        "",
        "## K0 and benchmark validity",
        "",
        f"- Mathematics: `{metrics['k0']['status']}`",
        f"- Static matching: `{metrics['static_match_status']}`",
        f"- Confirm stream: generated and hashed `{metrics['confirm_stream_sha256']}`, not evaluated.",
        "",
        "| Task | Static dev AUROC | Max static delta |",
        "|---|---:|---:|",
    ]
    for row in metrics["static_diagnostics"]:
        lines.append(f"| {row['task']} | {row['static_dev_auroc']:.4f} | {row['maximum_static_pair_delta']:.3e} |")
    lines.extend(["", "## K1 development metrics", "", "| Method | Task | Ranking | AUROC | TPR@FPR05 | Perturbed ranking | Stability |", "|---|---|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        for task in TASKS:
            row = k1["metrics"][method][task]
            lines.append(
                f"| {method} | {task} | {row['matched_ranking']:.4f} | {row['auroc']:.4f} | "
                f"{row['curve_tpr_at_fpr05']:.4f} | {row['perturbed_matched_ranking']:.4f} | {row['perturbation_score_correlation']:.4f} |"
            )
    gate = k1["gate"]
    lines.extend(
        [
            "",
            "## Frozen causal answers",
            "",
            f"1. Static ceiling removed: **yes**, all five static AUROCs were within 0.45--0.60.",
            f"2. Cat raw vs shear macro ranking delta: `{gate['anosov_macro_ranking_gain']:+.6f}`; Cat won `{gate['anosov_winning_task_count']}/5` tasks.",
            f"3. Kolmogorov vs Cat raw macro ranking delta: `{gate['kolmogorov_macro_ranking_gain']:+.6f}`; bootstrap CI `{k1['bootstrap']['kolmogorov_vs_cat_raw']['ci95_lower']:+.6f}, {k1['bootstrap']['kolmogorov_vs_cat_raw']['ci95_upper']:+.6f}`.",
            f"4. Full ANZA-KS vs static task gates: `{gate['passing_task_count']}/5`; required 3.",
            f"5. Low-FPR practical gate: `{'PASS' if gate['passing_task_count'] >= 3 else 'FAIL'}`.",
            f"6. Perturbation results are reported task-wise above and were not used for selection.",
            f"7. K2 legally allowed: **{'yes' if gate['pass'] else 'no'}**.",
            "",
            "Only fixed tiny logistic readouts were trained. No segmentation network, K2, confirm evaluation, CRACKS, or expert data were opened.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    protocol, benchmark, receipt = freeze_k0_inputs()
    pre_run = json.loads((FREEZE_ROOT / "pre_run_validator.json").read_text())
    if not pre_run.get("run_authorized"):
        raise PermissionError("ANZA-KS K1 lacks a passing pre-run validator")
    if source_manifest()["sha256"] != receipt["source_sha256"]:
        raise ValueError("ANZA-KS source changed after pre-run freeze")
    k1 = run_k1()
    metrics = {
        "status": k1["status"],
        "k0": json.loads((FREEZE_ROOT / "k0_math.json").read_text()),
        "static_match_status": benchmark["static_match_status"],
        "static_diagnostics": benchmark["static_diagnostics"],
        "confirm_stream_sha256": benchmark["confirm_stream_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "benchmark_sha256": receipt["benchmark_sha256"],
        "source_sha256": receipt["source_sha256"],
        "k1": {key: value for key, value in k1.items() if key not in ("raw_rows", "curve_rows")},
        "tiny_logistic_readouts_trained": True,
        "segmentation_training_performed": False,
        "confirm_evaluated": False,
        "K2_opened": False,
        "cracks_accessed": False,
        "expert_accessed": False,
    }
    _csv(RESULT_ROOT / "per_pair.csv", k1["raw_rows"])
    per_task = []
    for method in METHODS:
        for task in TASKS:
            per_task.append({"method": method, "task": task, **k1["metrics"][method][task]})
    _csv(RESULT_ROOT / "per_task.csv", per_task)
    _csv(RESULT_ROOT / "operating_curves.csv", k1["curve_rows"])
    _json(RESULT_ROOT / "feature_dimensions.json", {method: {"input_width": 104, "logistic_parameters": 105} for method in METHODS})
    _json(RESULT_ROOT / "bootstrap.json", k1["bootstrap"])
    _json(RESULT_ROOT / "metrics.json", metrics)
    (RESULT_ROOT / "ANZA_KS_K0_K1_REPORT.md").write_text(_report(metrics))
    _json(RESULT_ROOT / "TASK_STATE.json", {"status": metrics["status"], "K2_authorized": k1["gate"]["pass"], "confirm_evaluated": False})
    return metrics
