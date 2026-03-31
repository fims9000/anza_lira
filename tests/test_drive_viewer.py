from __future__ import annotations

import json
import numpy as np

from drive_viewer import DriveRunInfo, discover_drive_runs, metrics_from_prob_map, recommended_threshold_for_run


def test_discover_drive_runs_sorts_by_dice(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "checkpoint_best.pt").write_bytes(b"a")
    (run_b / "checkpoint_best.pt").write_bytes(b"b")
    (run_a / "metrics.json").write_text(json.dumps({"variant": "az_cat", "test_dice": 0.77}), encoding="utf-8")
    (run_b / "metrics.json").write_text(json.dumps({"variant": "baseline", "test_dice": 0.72}), encoding="utf-8")

    runs = discover_drive_runs(tmp_path)
    assert [run.name for run in runs] == ["run_a", "run_b"]
    assert runs[0].variant == "az_cat"
    assert runs[0].test_dice == 0.77


def test_metrics_from_prob_map_is_perfect_for_perfect_prediction():
    prob = [[0.01, 0.99], [0.98, 0.02]]
    target = [[0.0, 1.0], [1.0, 0.0]]
    valid = [[1.0, 1.0], [1.0, 1.0]]
    metrics = metrics_from_prob_map(
        prob_map=np.array(prob, dtype="float32"),
        target_mask=np.array(target, dtype="float32"),
        valid_mask=np.array(valid, dtype="float32"),
        threshold=0.5,
    )
    assert metrics["dice"] > 0.999
    assert metrics["iou"] > 0.999
    assert metrics["accuracy"] > 0.999


def test_recommended_threshold_for_run_reads_threshold_sweep(tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").write_bytes(b"x")
    (run_dir / "threshold_sweep.json").write_text(json.dumps({"recommended_threshold": 0.55}), encoding="utf-8")

    run = DriveRunInfo(
        name="run_x",
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint_best.pt",
        metrics_path=None,
        metrics={},
        variant="az_cat",
        test_dice=None,
        test_iou=None,
        test_recall=None,
        test_precision=None,
        num_parameters=None,
        approx_gmacs_per_forward=None,
    )
    assert recommended_threshold_for_run(run, default_threshold=0.6) == 0.55


def test_recommended_threshold_for_run_falls_back_to_metrics_selected_threshold(tmp_path):
    run_dir = tmp_path / "run_y"
    run_dir.mkdir()
    (run_dir / "checkpoint_best.pt").write_bytes(b"x")
    (run_dir / "metrics.json").write_text(json.dumps({"selected_threshold": 0.75}), encoding="utf-8")

    run = DriveRunInfo(
        name="run_y",
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoint_best.pt",
        metrics_path=run_dir / "metrics.json",
        metrics={"selected_threshold": 0.75},
        variant="az_thesis",
        test_dice=None,
        test_iou=None,
        test_recall=None,
        test_precision=None,
        num_parameters=None,
        approx_gmacs_per_forward=None,
    )
    assert recommended_threshold_for_run(run, default_threshold=0.6) == 0.75
