#!/usr/bin/env python3
"""Resumable deadline path for Setting A and the corrected synthetic audit.

This entrypoint deliberately does not import the corrected evaluator at module
load time.  It can therefore be installed and tested while the final evaluator
is still being implemented, without touching an active training process.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.anza_v2_deadline_common import (
    atomic_write_json,
    canonical_sha256,
    file_records,
    read_json,
    sha256_file,
    utc_now,
    verify_file_records,
)


STUDY_ROOT = PROJECT_ROOT / "results" / "anza_v2_study"
DEADLINE_ROOT = STUDY_ROOT / "deadline_20260817"
PHASE_ROOT = DEADLINE_ROOT / "phase_receipts"
ORIGINAL_RANGE = (0, 2000)
REPLACEMENT_RANGE = (2000, 4000)


def _setting_a_specs() -> tuple[Any, ...]:
    from cracks_experiment.matrix import setting_a_matrix

    return setting_a_matrix()


def _run_dir(spec: Any) -> Path:
    return STUDY_ROOT / "cracks" / "setting_a" / f"{spec.run_id}-{spec.run_hash}"


def create_pre_fix_snapshot(*, overwrite: bool = False) -> dict[str, Any]:
    """Record the immutable pre-fix state without reading expert metrics."""
    output = DEADLINE_ROOT / "PRE_FIX_SNAPSHOT.json"
    if output.exists() and not overwrite:
        existing = read_json(output)
        if existing.get("status") not in {"CAPTURED_BEFORE_EVALUATOR_FIX", "PRE_FIX_SNAPSHOT"}:
            raise RuntimeError(f"Invalid existing pre-fix snapshot: {output}")
        return {**existing, "action": "SKIP"}

    processes = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,args="],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    training = [line.strip() for line in processes if "anza_v2_study.py cracks-train" in line]
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    runs = []
    for spec in _setting_a_specs():
        status_path = _run_dir(spec) / "status.json"
        if status_path.exists():
            payload = read_json(status_path)
            state = payload.get("status", "UNKNOWN")
            epoch = payload.get("epoch", 0)
        else:
            state, epoch = "PENDING_NOT_STARTED", 0
        runs.append(
            {
                "run_id": spec.run_id,
                "expected_run_hash": spec.run_hash,
                "status": state,
                "epoch": epoch,
            }
        )
    protocol = STUDY_ROOT / "protocol.json"
    frozen = STUDY_ROOT / "synthetic" / "frozen_v2.json"
    payload = {
        "schema_version": 1,
        "status": "CAPTURED_BEFORE_EVALUATOR_FIX",
        "action": "RUN",
        "created_utc": utc_now(),
        "branch": branch,
        "git_status_short": status,
        "training_processes": training,
        "protocol_sha256": sha256_file(protocol),
        "synthetic_frozen_v2_sha256": sha256_file(frozen),
        "run_count_expected": len(runs),
        "run_count_complete": sum(item["status"] == "COMPLETE" for item in runs),
        "runs": runs,
        "expert_artifacts_read": False,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    atomic_write_json(output, payload)
    return payload


def audit_completed_runs() -> dict[str, Any]:
    """Verify all frozen Setting A runs and record checkpoint hashes."""
    records = []
    for spec in _setting_a_specs():
        run_dir = _run_dir(spec)
        status_path = run_dir / "status.json"
        checkpoint = run_dir / "checkpoint-last.pt"
        if not status_path.exists() or not checkpoint.exists():
            raise RuntimeError(f"Setting A run is not complete: {spec.run_id}")
        status = read_json(status_path)
        history = status.get("history", [])
        if (
            status.get("status") != "COMPLETE"
            or status.get("epoch") != 20
            or status.get("epoch_budget") != 20
            or status.get("run_hash") != spec.run_hash
            or status.get("expert_scores_used") is not False
            or status.get("checkpoint_reload") != "PASS"
            or [row.get("epoch") for row in history] != list(range(1, 21))
        ):
            raise RuntimeError(f"Invalid completed Setting A run: {spec.run_id}")
        records.append(
            {
                "run_id": spec.run_id,
                "model": spec.model,
                "seed": spec.seed,
                "comparison_family": spec.comparison_family,
                "run_hash": spec.run_hash,
                "status_sha256": sha256_file(status_path),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "expert_scores_used": False,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "run_count": len(records),
        "epoch_count_each": 20,
        "expert_scores_used": False,
        "records": records,
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    atomic_write_json(DEADLINE_ROOT / "RUNS_15_AUDIT.json", payload)
    _write_training_index(records)
    return payload


def _write_training_index(records: list[dict[str, Any]]) -> None:
    history_path = DEADLINE_ROOT / "training_history.csv"
    index_path = DEADLINE_ROOT / "RUN_INDEX_FINAL.tsv"
    DEADLINE_ROOT.mkdir(parents=True, exist_ok=True)
    history_fields = (
        "run_id", "run_hash", "seed", "epoch", "train_loss",
        "heldout_crop_dice_at_0_5", "checkpoint_sha256", "status", "expert_scores_used",
    )
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            status = read_json(_run_dir(type("Spec", (), record)) / "status.json")
            for row in status["history"]:
                writer.writerow(
                    {
                        "run_id": record["run_id"],
                        "run_hash": record["run_hash"],
                        "seed": record["seed"],
                        "epoch": row["epoch"],
                        "train_loss": row["train_loss"],
                        "heldout_crop_dice_at_0_5": row["heldout_crop_dice_at_0_5"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "status": "COMPLETE",
                        "expert_scores_used": "false",
                    }
                )
    fields = (
        "run_id", "model", "seed", "comparison_family", "run_hash", "checkpoint_sha256",
        "checkpoint_size_bytes", "expert_scores_used",
    )
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: record[key] for key in fields} for record in records])


def _invoke(module_name: str, names: Iterable[str], **context: Any) -> Any:
    """Import a late-arriving evaluator/reporting module only when its phase runs."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise RuntimeError(f"Deadline phase dependency is not implemented yet: {module_name}") from error
        raise
    function = next((getattr(module, name, None) for name in names if callable(getattr(module, name, None))), None)
    if function is None:
        raise RuntimeError(f"{module_name} does not expose any of: {', '.join(names)}")
    parameters = inspect.signature(function).parameters
    kwargs = {name: value for name, value in context.items() if name in parameters}
    return function(**kwargs)


def _invoke_first(modules: Iterable[str], names: Iterable[str], **context: Any) -> Any:
    failures = []
    for module in modules:
        try:
            return _invoke(module, names, **context)
        except RuntimeError as error:
            failures.append(str(error))
    raise RuntimeError("; ".join(failures))


def _artifact_paths(result: Any) -> list[Path]:
    if not isinstance(result, dict):
        return []
    values = result.get("artifacts", [])
    if isinstance(values, (str, Path)):
        values = [values]
    return [Path(value) if Path(value).is_absolute() else PROJECT_ROOT / value for value in values]


def _phase_input_hash(name: str, input_paths: Iterable[Path]) -> str:
    records = file_records(input_paths, base=PROJECT_ROOT)
    return canonical_sha256({"phase": name, "inputs": records})


def run_phase(
    name: str,
    runner: Callable[[], Any],
    *,
    input_paths: Iterable[Path],
    required_outputs: Iterable[Path],
) -> dict[str, Any]:
    """Run a phase once per exact input/output hash contract."""
    input_paths = tuple(Path(path) for path in input_paths)
    required_outputs = tuple(Path(path) for path in required_outputs)
    input_hash = _phase_input_hash(name, input_paths)
    receipt_path = PHASE_ROOT / f"{name}.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") == "COMPLETE"
            and receipt.get("input_sha256") == input_hash
            and verify_file_records(receipt.get("outputs", []), base=PROJECT_ROOT)
        ):
            return {**receipt, "action": "SKIP"}
    result = runner()
    outputs = list(required_outputs)
    outputs.extend(path for path in _artifact_paths(result) if path not in outputs)
    output_records = file_records(outputs, base=PROJECT_ROOT)
    receipt = {
        "schema_version": 1,
        "phase": name,
        "status": "COMPLETE",
        "action": "RUN",
        "finished_utc": utc_now(),
        "input_sha256": input_hash,
        "outputs": output_records,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def _run_corrected_tests() -> dict[str, Any]:
    tests = (
        PROJECT_ROOT / "tests" / "test_synthetic_evaluator_corrected.py",
        PROJECT_ROOT / "tests" / "test_synthetic_deadline_evaluation_runner.py",
    )
    missing = [path for path in tests if not path.exists()]
    if missing:
        raise RuntimeError(f"Corrected evaluator tests are not implemented yet: {missing}")
    log = DEADLINE_ROOT / "corrected_evaluator_tests.log"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(str(path.relative_to(PROJECT_ROOT)) for path in tests),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Corrected evaluator tests failed; see {log}")
    return {"artifacts": [log]}


def _run_crowd_validation() -> dict[str, Any]:
    from cracks_experiment.validation import freeze_setting_a_thresholds, run_setting_a_validation

    training_root = STUDY_ROOT / "cracks" / "setting_a"
    artifacts = []
    for spec in _setting_a_specs():
        run_setting_a_validation(spec, training_root)
        artifacts.extend(
            [
                _run_dir(spec) / "crowd_validation.json",
                _run_dir(spec) / "crowd_validation_sections.csv",
            ]
        )
    freeze_setting_a_thresholds(training_root)
    artifacts.append(training_root / "threshold_freeze.json")
    return {"artifacts": artifacts}


def _run_expert_evaluation() -> dict[str, Any]:
    from cracks_experiment.evaluation import (
        finalize_setting_a_expert_evaluation,
        run_setting_a_expert_evaluation,
    )

    training_root = STUDY_ROOT / "cracks" / "setting_a"
    expert_root = STUDY_ROOT / "cracks" / "setting_a_expert"
    artifacts = []
    for spec in _setting_a_specs():
        run_setting_a_expert_evaluation(spec, training_root, expert_root)
        artifacts.extend(
            [
                expert_root / f"{spec.run_id}-{spec.run_hash}.csv",
                expert_root / f"{spec.run_id}-{spec.run_hash}.json",
            ]
        )
    finalize_setting_a_expert_evaluation(training_root, expert_root)
    artifacts.append(expert_root / "complete.json")
    return {"artifacts": artifacts}


def _corrected_context() -> dict[str, Any]:
    return {
        "study_root": STUDY_ROOT,
        "output_root": STUDY_ROOT / "synthetic" / "evaluator_audit",
        "deadline_root": DEADLINE_ROOT,
        "original_start": ORIGINAL_RANGE[0],
        "original_stop": ORIGINAL_RANGE[1],
        "replacement_start": REPLACEMENT_RANGE[0],
        "replacement_stop": REPLACEMENT_RANGE[1],
    }


def _synthetic_freeze_inputs() -> tuple[dict[str, str], dict[str, float]]:
    from synthetic.experiment_matrix import development_matrix

    selected = {"B0", "B1", "C0", "C3"}
    checkpoint_hashes: dict[str, str] = {}
    visible_thresholds: dict[str, float] = {}
    for spec in development_matrix():
        if spec.candidate_id not in selected:
            continue
        run_name = f"{spec.candidate_id}-{spec.run_hash}"
        checkpoint = STUDY_ROOT / "synthetic" / "development" / run_name / "checkpoint-last.pt"
        validation = read_json(STUDY_ROOT / "synthetic" / "validation" / f"{run_name}.json")
        checkpoint_hashes[spec.candidate_id] = sha256_file(checkpoint)
        visible_thresholds[spec.candidate_id] = float(validation["selected_visible_threshold"])
    if set(checkpoint_hashes) != selected or set(visible_thresholds) != selected:
        raise RuntimeError("Corrected synthetic freeze requires frozen B0/B1/C0/C3 inputs")
    return checkpoint_hashes, visible_thresholds


def _freeze_corrected_evaluator() -> dict[str, Any]:
    from synthetic.evaluator_audit import freeze_corrected_evaluator

    checkpoint_hashes, thresholds = _synthetic_freeze_inputs()
    result = freeze_corrected_evaluator(
        STUDY_ROOT,
        model_checkpoint_hashes=checkpoint_hashes,
        visible_thresholds=thresholds,
    )
    return {**result, "artifacts": [STUDY_ROOT / "synthetic" / "replacement_confirmation" / "freeze.json"]}


def _compute_corrected(kind: str) -> dict[str, Any]:
    names = {
        "validation": ("compute_validation_audit", "evaluate_corrected_validation"),
        "legacy": ("compute_legacy_reanalysis", "evaluate_legacy_reanalysis"),
        "replacement": ("compute_replacement_confirmation", "evaluate_replacement_confirmation"),
    }[kind]
    result = _invoke_first(
        ("synthetic.deadline_evaluation_runner", "synthetic.evaluator_audit"),
        names,
        **_corrected_context(),
    )
    if not isinstance(result, dict) or not result:
        raise RuntimeError(f"Corrected {kind} computation returned no machine evidence")
    return result


def _run_corrected_validation() -> dict[str, Any]:
    from synthetic.evaluator_audit import run_validation_audit

    result = run_validation_audit(STUDY_ROOT, audit=_compute_corrected("validation"))
    return {**result, "artifacts": [STUDY_ROOT / "synthetic" / "evaluator_audit" / "validation_audit.json"]}


def _run_legacy_reanalysis() -> dict[str, Any]:
    from synthetic.evaluator_audit import run_legacy_test_reanalysis

    result = run_legacy_test_reanalysis(STUDY_ROOT, reanalysis=_compute_corrected("legacy"))
    return {
        **result,
        "artifacts": [STUDY_ROOT / "synthetic" / "evaluator_audit" / "legacy_test_reanalysis" / "summary.json"],
    }


def _run_replacement() -> dict[str, Any]:
    from synthetic.evaluator_audit import run_replacement_confirmation

    computed = _compute_corrected("replacement")
    if int(computed.get("sample_count", -1)) != 2000:
        raise RuntimeError("Replacement computation must contain exactly 2000 previously unopened samples")
    synthetic_rows = computed.get("synthetic_corrected_rows", computed.get("synthetic_rows"))
    if not isinstance(synthetic_rows, list) or not synthetic_rows:
        raise RuntimeError("Replacement computation did not return corrected synthetic_rows")
    computed = {**computed, "synthetic_rows": synthetic_rows}
    result = run_replacement_confirmation(STUDY_ROOT, confirmation=computed)
    return {**result, "artifacts": [STUDY_ROOT / "synthetic" / "replacement_confirmation" / "summary.json"]}


def _write_synthetic_gate_audit() -> dict[str, Any]:
    replacement = read_json(STUDY_ROOT / "synthetic" / "replacement_confirmation" / "summary.json")
    if replacement.get("scientific_result") is not True:
        raise RuntimeError("Replacement confirmation is not a full scientific result")
    c3 = replacement.get("models", {}).get("C3", {})
    metrics = c3.get("metrics", {}) if isinstance(c3, dict) else {}
    required = (
        "route_excess_over_chance",
        "route_excess_over_chance_ci95_low",
        "route_excess_over_chance_ci95_high",
    )
    if c3.get("model") != "anza_v2b" or any(metrics.get(key) is None for key in required):
        raise RuntimeError("Replacement summary lacks frozen C3 routing evidence")
    estimate, low, high = (float(metrics[key]) for key in required)
    if not all(np.isfinite(value) for value in (estimate, low, high)) or low > high:
        raise RuntimeError("Replacement C3 routing evidence is non-finite or unordered")
    verdict = (
        "SUPPORTED_ABOVE_CHANCE"
        if low > 0
        else "NEGATIVE" if high < 0 else "NOT_ESTABLISHED"
    )
    evidence = {
        "verdict": verdict,
        "model": "anza_v2b",
        "route_excess_over_chance": estimate,
        "route_excess_over_chance_ci95_low": low,
        "route_excess_over_chance_ci95_high": high,
        "bootstrap_unit": "synthetic_sample",
        "source": "replacement_confirmation.models.C3.metrics",
    }
    if verdict not in {"SUPPORTED_ABOVE_CHANCE", "NOT_ESTABLISHED", "NEGATIVE"}:
        raise RuntimeError(f"Invalid corrected mechanism verdict: {verdict}")
    false_bridge_rates = {}
    for candidate_id, model in replacement.get("models", {}).items():
        rows = model.get("rows", [])
        false_count = sum(int(row.get("false_bridge_count", 0)) for row in rows)
        negative_count = sum(int(row.get("negative_gap_count", 0)) for row in rows)
        false_bridge_rates[candidate_id] = false_count / negative_count if negative_count else 0.0
    saturated = bool(false_bridge_rates) and all(
        np.isclose(rate, 1.0) for rate in false_bridge_rates.values()
    )
    false_bridge = {
        "status": (
            "FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE"
            if saturated
            else "FALSE_BRIDGE_ENDPOINT_RETAINS_DISCRIMINATIVE_RANGE"
        ),
        "method_rates": false_bridge_rates,
        "denominator": "negative_gap_count",
        "threshold_selection_permitted": False,
    }
    payload = {
        "status": "COMPLETE",
        "legacy_gate_status": read_json(STUDY_ROOT / "synthetic" / "frozen_v2.json")["quality_gate"],
        "legacy_gate_validity": "PARTIALLY_INVALIDATED",
        "invalidated_fields": [
            "B0/B1/C0 model-labelled branch pairing from generator geometry",
            "C3 threshold-0.5 primary pairing readout",
        ],
        "still_valid_fields": [
            "segmentation and trace metrics",
            "checkpoint provenance",
            "test-open provenance",
        ],
        "corrected_mechanism_evidence": evidence,
        "false_bridge_verdict": false_bridge,
    }
    output = DEADLINE_ROOT / "SYNTHETIC_GATE_AUDIT.json"
    atomic_write_json(output, payload)
    return {**payload, "artifacts": [output]}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _deadline_result_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    expert_root = STUDY_ROOT / "cracks" / "setting_a_expert"
    main_rows: list[dict[str, str]] = []
    ablation_rows: list[dict[str, str]] = []
    for spec in _setting_a_specs():
        rows = [
            row for row in _read_csv(expert_root / f"{spec.run_id}-{spec.run_hash}.csv")
            if row["policy"] == "paper_like"
        ]
        if spec.comparison_family == "main":
            main_rows.extend(rows)
        if spec.run_id == "v2_s42":
            ablation_rows.extend({**row, "run_id": "v2_full_s42"} for row in rows)
        elif spec.comparison_family == "ablation":
            ablation_rows.extend(rows)
    replacement = read_json(STUDY_ROOT / "synthetic" / "replacement_confirmation" / "summary.json")
    synthetic_rows = replacement.get("synthetic_rows")
    if not isinstance(synthetic_rows, list) or not synthetic_rows:
        raise RuntimeError("Corrected synthetic rows missing from replacement confirmation")
    corrected_rows = []
    models = replacement.get("models", {})
    for original in synthetic_rows:
        row = dict(original)
        candidate_id = row.get("candidate_id")
        model_summary = models.get(candidate_id)
        if isinstance(model_summary, dict):
            raw_rows = model_summary.get("rows", [])
            false_count = sum(int(item.get("false_bridge_count", 0)) for item in raw_rows)
            negative_count = sum(int(item.get("negative_gap_count", 0)) for item in raw_rows)
            positive_rows = [
                item for item in raw_rows if int(item.get("positive_gap_count", 0)) > 0
            ]
            row["false_bridge_count"] = false_count
            row["negative_gap_count"] = negative_count
            row["false_bridge_rate"] = false_count / negative_count if negative_count else 0.0
            row["positive_gap_count"] = sum(
                int(item.get("positive_gap_count", 0)) for item in raw_rows
            )
            row["gap_recovery_rate"] = (
                float(np.mean([float(item["gap_recovery_rate"]) for item in positive_rows]))
                if positive_rows
                else 1.0
            )
        corrected_rows.append(row)
    return main_rows, ablation_rows, corrected_rows


def _run_deadline_statistics() -> dict[str, Any]:
    from cracks_experiment.deadline_reporting import build_deadline_statistics

    main, ablations, synthetic = _deadline_result_rows()
    result = build_deadline_statistics(
        DEADLINE_ROOT,
        expert_rows=main,
        ablation_rows=ablations,
        synthetic_rows=synthetic,
    )
    output = DEADLINE_ROOT / "statistics.json"
    atomic_write_json(output, result)
    return {
        **result,
        "artifacts": [
            output,
            DEADLINE_ROOT / "tables" / "main_cracks.csv",
            DEADLINE_ROOT / "tables" / "paired_comparisons.csv",
            DEADLINE_ROOT / "tables" / "ablations.csv",
            DEADLINE_ROOT / "tables" / "synthetic_corrected.csv",
            DEADLINE_ROOT / "raw_per_section.csv",
            DEADLINE_ROOT / "QUALITATIVE_SELECTION.json",
        ],
    }


def _qualitative_panels() -> dict[int, dict[str, Any]]:
    import torch
    from cracks_experiment.figures import _load_real_model
    from cracks_experiment.human import _normalized_image
    from cracks_experiment.validation import tiled_probability
    from datasets.cracks import load_rgb_mask, load_section_image, map_mask_rgb

    selected = read_json(DEADLINE_ROOT / "QUALITATIVE_SELECTION.json")["selected_section_ids"]
    specs = {
        spec.run_id: spec for spec in _setting_a_specs()
        if spec.run_id in {"unet_s42", "v1_s42", "v2_s42"}
    }
    setting_root = STUDY_ROOT / "cracks" / "setting_a"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: _load_real_model(spec, setting_root, device) for name, spec in specs.items()}
    thresholds = {
        name: float(read_json(_run_dir(spec) / "crowd_validation.json")["selected_threshold"])
        for name, spec in specs.items()
    }
    panels: dict[int, dict[str, Any]] = {}
    for section_id in selected:
        image = load_section_image(PROJECT_ROOT / "data" / "cracks" / "images" / f"section_{section_id:03d}.png")
        expert, valid, _ = map_mask_rgb(
            load_rgb_mask(PROJECT_ROOT / "data" / "cracks" / "annotations" / "expert" / f"section_{section_id:03d}.png"),
            "paper_like",
        )
        normalized = _normalized_image(section_id)
        row: dict[str, Any] = {"input": image, "expert": expert * valid}
        for run_id, label in (("unet_s42", "unet"), ("v1_s42", "anza_v1"), ("v2_s42", "anza_v2b")):
            row[label] = tiled_probability(models[run_id], normalized).numpy()[:255, :701] >= thresholds[run_id]
        panels[int(section_id)] = row
    return panels


def _run_deadline_figures() -> dict[str, Any]:
    from cracks_experiment.deadline_reporting import generate_deadline_figures

    result = generate_deadline_figures(DEADLINE_ROOT, qualitative_panels=_qualitative_panels())
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"Deadline figures incomplete: {result}")
    manifest = DEADLINE_ROOT / "figures" / "manifest.json"
    atomic_write_json(manifest, result)
    artifacts = [manifest]
    artifacts.extend(
        DEADLINE_ROOT / "figures" / f"{stem}.{suffix}"
        for stem in ("fig_cracks_main", "fig_cracks_examples", "fig_ablation")
        for suffix in ("png", "svg", "pdf")
    )
    optional = DEADLINE_ROOT / "figures" / "fig_synthetic_mechanism.png"
    if optional.exists():
        artifacts.extend(optional.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    return {**result, "artifacts": artifacts}


def _run_deadline_evidence() -> dict[str, Any]:
    from cracks_experiment.deadline_reporting import build_deadline_evidence

    result = build_deadline_evidence(DEADLINE_ROOT)
    readme = DEADLINE_ROOT / "README_FIRST.md"
    readme.write_text(
        "# Read this first\n\n"
        "This package is the deadline-scoped Setting A result, not completion of the full CRACKS study. "
        "Start with `DEADLINE_REPORT.md`, then verify `validator_result.json` and `SHA256SUMS.txt`.\n"
    )
    return {
        **result,
        "artifacts": [
            DEADLINE_ROOT / "THESIS_NUMBERS.json",
            DEADLINE_ROOT / "THESIS_EVIDENCE.md",
            DEADLINE_ROOT / "DEADLINE_REPORT.md",
            DEADLINE_ROOT / "DEADLINE_SCOPE.md",
            readme,
        ],
    }


def _write_scope() -> dict[str, Any]:
    path = DEADLINE_ROOT / "DEADLINE_SCOPE.md"
    text = (
        "# Deadline scope\n\n"
        "Setting A plus the corrected synthetic evaluator and replacement confirmation are included. "
        "Settings B/C are deferred as `NOT_RUN_DEADLINE_SCOPE` and are not used in submitted claims.\n"
    )
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {"artifacts": [path]}


def _package_arcname(path: Path) -> Path:
    """Return a stable in-archive name for study and explicitly bundled docs."""

    resolved = path.resolve()
    study_root = STUDY_ROOT.resolve()
    project_root = PROJECT_ROOT.resolve()
    if resolved.is_relative_to(study_root):
        return resolved.relative_to(study_root)
    if resolved.is_relative_to(project_root):
        return resolved.relative_to(project_root)
    raise ValueError(f"Package candidate is outside the project: {path}")


def _package() -> dict[str, Any]:
    required = [
        DEADLINE_ROOT / "README_FIRST.md",
        DEADLINE_ROOT / "DEADLINE_SCOPE.md",
        DEADLINE_ROOT / "SYNTHETIC_GATE_AUDIT.json",
        DEADLINE_ROOT / "training_history.csv",
        DEADLINE_ROOT / "RUN_INDEX_FINAL.tsv",
        DEADLINE_ROOT / "DEADLINE_REPORT.md",
        DEADLINE_ROOT / "THESIS_NUMBERS.json",
        DEADLINE_ROOT / "THESIS_EVIDENCE.md",
        DEADLINE_ROOT / "raw_per_section.csv",
        STUDY_ROOT / "cracks" / "setting_a" / "threshold_freeze.json",
        DEADLINE_ROOT / "validator_result.json",
    ]
    required.extend((DEADLINE_ROOT / "tables" / name) for name in (
        "main_cracks.csv", "paired_comparisons.csv", "ablations.csv", "synthetic_corrected.csv"
    ))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = set(required)
    for root in (
        STUDY_ROOT / "cracks" / "setting_a_expert",
        STUDY_ROOT / "synthetic" / "evaluator_audit",
        STUDY_ROOT / "synthetic" / "replacement_confirmation",
        DEADLINE_ROOT / "figures",
    ):
        if root.exists():
            candidates.update(path for path in root.rglob("*") if path.is_file())
    candidates.update(path for path in DEADLINE_ROOT.glob("*.json") if path.is_file())
    candidates.update(path for path in DEADLINE_ROOT.glob("*.md") if path.is_file())
    audit_doc = PROJECT_ROOT / "docs" / "research" / "SYNTHETIC_EVALUATOR_AUDIT_20260817.md"
    if audit_doc.exists():
        candidates.add(audit_doc)
    candidates = {path for path in candidates if path.suffix != ".pt" and path.name != "SHA256SUMS.txt"}
    sums = DEADLINE_ROOT / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{sha256_file(path)}  {_package_arcname(path)}\n" for path in sorted(candidates))
    )
    candidates.add(sums)
    package_root = DEADLINE_ROOT / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().replace("-", "").replace(":", "").split(".")[0].replace("+0000", "Z")
    package = package_root / f"ANZA_LIRA_DEADLINE_FINAL_20260817_{timestamp}.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, arcname=str(_package_arcname(path)))
    return {"artifacts": [package, sums], "package": str(package), "sha256": sha256_file(package)}


def _phases() -> list[tuple[str, Callable[[], Any], tuple[Path, ...], tuple[Path, ...]]]:
    snapshot = DEADLINE_ROOT / "PRE_FIX_SNAPSHOT.json"
    runs_audit = DEADLINE_ROOT / "RUNS_15_AUDIT.json"
    test_log = DEADLINE_ROOT / "corrected_evaluator_tests.log"
    corrected_freeze = STUDY_ROOT / "synthetic" / "replacement_confirmation" / "freeze.json"
    validation_audit = STUDY_ROOT / "synthetic" / "evaluator_audit" / "validation_audit.json"
    legacy_audit = STUDY_ROOT / "synthetic" / "evaluator_audit" / "legacy_test_reanalysis" / "summary.json"
    replacement_freeze = STUDY_ROOT / "synthetic" / "replacement_confirmation" / "freeze.json"
    replacement_summary = STUDY_ROOT / "synthetic" / "replacement_confirmation" / "summary.json"
    threshold_freeze = STUDY_ROOT / "cracks" / "setting_a" / "threshold_freeze.json"
    expert_receipt = STUDY_ROOT / "cracks" / "setting_a_expert" / "complete.json"
    gate_audit = DEADLINE_ROOT / "SYNTHETIC_GATE_AUDIT.json"
    statistics = DEADLINE_ROOT / "statistics.json"
    figures = DEADLINE_ROOT / "figures" / "manifest.json"
    evidence = DEADLINE_ROOT / "DEADLINE_REPORT.md"
    scope = DEADLINE_ROOT / "DEADLINE_SCOPE.md"
    validator = DEADLINE_ROOT / "validator_result.json"
    return [
        ("snapshot", create_pre_fix_snapshot, (STUDY_ROOT / "protocol.json", STUDY_ROOT / "synthetic" / "frozen_v2.json"), (snapshot,)),
        ("runs_15_audit", audit_completed_runs, tuple(_run_dir(spec) / "status.json" for spec in _setting_a_specs()), (runs_audit, DEADLINE_ROOT / "training_history.csv", DEADLINE_ROOT / "RUN_INDEX_FINAL.tsv")),
        (
            "corrected_evaluator_tests",
            _run_corrected_tests,
            (
                PROJECT_ROOT / "tests" / "test_synthetic_evaluator_corrected.py",
                PROJECT_ROOT / "tests" / "test_synthetic_deadline_evaluation_runner.py",
            ),
            (test_log,),
        ),
        ("corrected_evaluator_freeze", _freeze_corrected_evaluator, (test_log, STUDY_ROOT / "synthetic" / "frozen_v2.json"), (corrected_freeze,)),
        ("corrected_validation_audit", _run_corrected_validation, (corrected_freeze,), (validation_audit,)),
        ("legacy_posthoc_reanalysis", _run_legacy_reanalysis, (corrected_freeze, STUDY_ROOT / "synthetic" / "test" / "test_open_receipt.json"), (legacy_audit,)),
        ("replacement_confirmation", _run_replacement, (corrected_freeze, validation_audit, legacy_audit), (replacement_summary,)),
        ("synthetic_gate_audit", _write_synthetic_gate_audit, (replacement_summary, PROJECT_ROOT / "scripts" / "anza_v2_deadline_finalize.py"), (gate_audit,)),
        ("crowd_threshold_freeze", _run_crowd_validation, (runs_audit,), (threshold_freeze,)),
        ("setting_a_expert", _run_expert_evaluation, (threshold_freeze,), (expert_receipt,)),
        ("deadline_statistics", _run_deadline_statistics, (expert_receipt, replacement_summary, PROJECT_ROOT / "scripts" / "anza_v2_deadline_finalize.py"), (statistics, DEADLINE_ROOT / "tables" / "main_cracks.csv", DEADLINE_ROOT / "tables" / "paired_comparisons.csv", DEADLINE_ROOT / "tables" / "ablations.csv", DEADLINE_ROOT / "tables" / "synthetic_corrected.csv")),
        ("deadline_figures", _run_deadline_figures, (statistics,), (figures, DEADLINE_ROOT / "figures" / "fig_cracks_main.png", DEADLINE_ROOT / "figures" / "fig_cracks_examples.png", DEADLINE_ROOT / "figures" / "fig_ablation.png")),
        ("deadline_scope", _write_scope, (snapshot,), (scope,)),
        ("deadline_evidence", _run_deadline_evidence, (statistics, figures, scope, gate_audit), (evidence, DEADLINE_ROOT / "README_FIRST.md", DEADLINE_ROOT / "THESIS_NUMBERS.json", DEADLINE_ROOT / "THESIS_EVIDENCE.md")),
        ("deadline_validator", lambda: subprocess.run([sys.executable, "scripts/validate_anza_v2_deadline.py"], cwd=PROJECT_ROOT, check=True), (evidence, PROJECT_ROOT / "scripts" / "validate_anza_v2_deadline.py"), (validator,)),
        ("deadline_package", _package, (validator, PROJECT_ROOT / "scripts" / "anza_v2_deadline_finalize.py"), (DEADLINE_ROOT / "SHA256SUMS.txt",)),
    ]


def run_deadline(*, stop_after: str | None = None) -> int:
    for name, runner, inputs, outputs in _phases():
        print(f"phase={name} status=STARTED", flush=True)
        receipt = run_phase(name, runner, input_paths=inputs, required_outputs=outputs)
        print(f"phase={name} status={receipt['status']} action={receipt['action']}", flush=True)
        if stop_after == name:
            break
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("snapshot", "audit-runs", "full"), default="full")
    parser.add_argument("--stop-after", choices=tuple(name for name, *_ in _phases()))
    args = parser.parse_args()
    if args.command == "snapshot":
        result = create_pre_fix_snapshot()
        print(f"PRE-FIX SNAPSHOT: {result['status']} ({result['action']})")
        return 0
    if args.command == "audit-runs":
        result = audit_completed_runs()
        print(f"SETTING A RUN AUDIT: {result['status']} ({result['run_count']}/15)")
        return 0
    return run_deadline(stop_after=args.stop_after)


if __name__ == "__main__":
    raise SystemExit(main())
