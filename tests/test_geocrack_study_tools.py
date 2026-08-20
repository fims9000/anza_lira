from __future__ import annotations

import math
import inspect
from pathlib import Path
import subprocess
import csv
import json

import pandas as pd

from scripts.geocrack_study import (
    RUN_MATRIX,
    _run_config_hash,
    cluster_bootstrap_delta,
    cluster_bootstrap_from_patch_frame,
    dry_run_matrix,
    resolve_run_action,
    evaluate_run,
    build_traces_for_run,
)
from scripts.validate_geocrack_study import all_finite, required_run_keys


def test_cluster_bootstrap_aggregates_paired_sources_deterministically() -> None:
    baseline = {"source_a": 0.5, "source_b": 0.6, "unpaired": 0.1}
    az = {"source_a": 0.7, "source_b": 0.7}
    first = cluster_bootstrap_delta(baseline, az, replicates=2000, seed=2026)
    second = cluster_bootstrap_delta(baseline, az, replicates=2000, seed=2026)
    assert first == second
    assert first["source_count"] == 2
    assert math.isclose(first["mean_delta_az_minus_baseline"], 0.15)
    assert first["ci95_low"] <= first["mean_delta_az_minus_baseline"] <= first["ci95_high"]


def test_patch_dataframe_bootstrap_resamples_sources_not_rows() -> None:
    rows = []
    for model, source, patch_count, value in (
        ("baseline", "source_many", 100, 0.0),
        ("az_thesis", "source_many", 100, 1.0),
        ("baseline", "source_one", 1, 1.0),
        ("az_thesis", "source_one", 1, 0.0),
    ):
        rows.extend(
            {"model": model, "seed": 42, "source_image_id": source, "dice": value}
            for _ in range(patch_count)
        )
    result = cluster_bootstrap_from_patch_frame(pd.DataFrame(rows), metric="dice", replicates=1000)
    assert result["resampling_unit"] == "source_image_id"
    assert result["source_count"] == 2
    assert result["patch_row_count"] == 202
    assert result["mean_delta_az_minus_baseline"] == 0.0


def test_run_config_hash_changes_with_scientific_inputs_not_resume_path() -> None:
    cfg = {"epochs": 30, "batch_size": 8, "resume_checkpoint": "old.pt", "run_name": "a"}
    original = _run_config_hash(cfg, "baseline", 42, "split-a")
    cfg["resume_checkpoint"] = "new.pt"
    cfg["run_name"] = "b"
    assert _run_config_hash(cfg, "baseline", 42, "split-a") == original
    assert _run_config_hash({**cfg, "epochs": 31}, "baseline", 42, "split-a") != original
    assert _run_config_hash(cfg, "baseline", 42, "split-b") != original


def test_recursive_finiteness_rejects_nan_and_infinity() -> None:
    assert all_finite({"a": [1, 2.5, {"b": None}]})
    assert not all_finite({"a": float("nan")})
    assert not all_finite({"a": float("inf")})


def test_required_run_matrix_is_exact() -> None:
    keys = required_run_keys()
    assert len(keys) == 9
    assert ("baseline", 41) in keys
    assert ("az_thesis", 43) in keys
    assert ("attention_unet", 42) in keys


def test_dry_run_matrix_order_and_protocol_hash_are_exact(capsys) -> None:
    payload = dry_run_matrix(Path("configs/geocrack_small.yaml"))
    assert [(row["model"], row["seed"]) for row in payload["runs"]] == list(RUN_MATRIX)
    assert payload["run_count"] == 9
    assert payload["fairness"] == "PASS"
    assert len(payload["protocol_hash"]) == 64
    output = capsys.readouterr().out
    assert output.splitlines()[:9] == [f"{model} seed {seed}" for model, seed in RUN_MATRIX]


def test_resume_action_contract_without_training() -> None:
    metadata = {"status": "COMPLETE", "config_hash": "cfg", "split_hash": "split"}
    assert resolve_run_action(
        metadata,
        config_hash="cfg",
        split_hash="split",
        checkpoint_last=True,
        checkpoint_best=True,
        metrics_present=True,
    ) == "SKIP"
    interrupted = {**metadata, "status": "RUNNING"}
    assert resolve_run_action(
        interrupted, config_hash="cfg", split_hash="split", checkpoint_last=True
    ) == "RESUME"
    assert resolve_run_action(
        interrupted, config_hash="changed", split_hash="split", checkpoint_last=True
    ) == "NEW_RUN_ID"
    assert resolve_run_action(
        interrupted, config_hash="cfg", split_hash="changed", checkpoint_last=True
    ) == "NEW_RUN_ID"


def test_linux_launcher_is_thin_and_executes_dry_run() -> None:
    script = Path("scripts/run_geocrack_full_study.sh")
    text = script.read_text(encoding="utf-8")
    assert "geocrack_study.py full" in text
    result = subprocess.run(["bash", str(script), "--dry-run"], text=True, capture_output=True, check=True)
    assert "baseline seed 41" in result.stdout
    assert "attention_unet seed 42" in result.stdout
    assert "PROTOCOL FAIRNESS: PASS" in result.stdout


def test_evaluation_and_trace_stages_are_code_independent() -> None:
    evaluation_source = inspect.getsource(evaluate_run)
    trace_source = inspect.getsource(build_traces_for_run)
    assert "compute_trace_metrics" not in evaluation_source
    assert "traces_to_geojson" not in evaluation_source
    assert "compute_trace_metrics" in trace_source
    assert "traces_to_geojson" in trace_source


def test_trace_stage_rebuilds_geojson_from_saved_arrays_without_model(tmp_path: Path) -> None:
    import numpy as np

    run_dir = tmp_path / "runs" / "baseline_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "COMPLETE", "model": "baseline", "seed": 42}), encoding="utf-8"
    )
    patch_id = "synthetic_straight_02_patch0"
    with (run_dir / "per_patch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "seed", "source_image_id", "patch_id", "threshold", "geometry_source", "dice"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "baseline",
                "seed": 42,
                "source_image_id": "synthetic_straight_02",
                "patch_id": patch_id,
                "threshold": 0.5,
                "geometry_source": "saved",
                "dice": 1.0,
            }
        )
    artifact_dir = tmp_path / "artifacts" / run_dir.name
    artifact_dir.mkdir(parents=True)
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, 3:18] = True
    np.savez_compressed(
        artifact_dir / f"{patch_id}.npz",
        target=skeleton,
        predicted=skeleton,
        probability=skeleton.astype(float),
        pred_skeleton=skeleton,
        orientation=np.zeros_like(skeleton, dtype=float),
        coherence=np.ones_like(skeleton, dtype=float),
        anisotropy=np.zeros_like(skeleton, dtype=float),
    )
    traces_root = tmp_path / "traces"
    build_traces_for_run(run_dir, artifact_root=tmp_path / "artifacts", traces_root=traces_root)
    complete = json.loads((run_dir / "traces_complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "PASS"
    assert complete["model_loaded"] is False
    assert (traces_root / run_dir.name / f"{patch_id}.geojson").is_file()
