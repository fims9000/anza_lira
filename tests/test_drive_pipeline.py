from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import utils
from models import AZSOTAUNet, AZUNet, BaselineUNet
from models.azconv import AZConvConfig


def _write_rgb(path, height: int = 32, width: int = 40) -> None:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    image = np.stack(
        [
            (xx * 5) % 255,
            (yy * 7) % 255,
            ((xx + yy) * 3) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image).save(path)


def _write_mask(path, height: int = 32, width: int = 40, invert: bool = False) -> None:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:24, 14:26] = 255
    if invert:
        mask[:] = 255
    Image.fromarray(mask).save(path)


def _make_drive_tree(root) -> None:
    for split, prefix in (("training", "01"), ("training", "02"), ("test", "21")):
        split_dir = root / "DRIVE" / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "1st_manual").mkdir(parents=True, exist_ok=True)
        (split_dir / "mask").mkdir(parents=True, exist_ok=True)

        image_stem = f"{prefix}_{'training' if split == 'training' else 'test'}"
        _write_rgb(split_dir / "images" / f"{image_stem}.tif")
        _write_mask(split_dir / "1st_manual" / f"{prefix}_manual1.gif")
        _write_mask(split_dir / "mask" / f"{image_stem}_mask.gif", invert=True)


def test_drive_dataloaders_build_and_batch(tmp_path):
    _make_drive_tree(tmp_path / "data")
    cfg = {
        "dataset": "drive",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
    }

    train_loader, val_loader, test_loader, in_channels, num_outputs, task = utils.build_dataloaders(cfg)
    assert task == "segmentation"
    assert in_channels == 3
    assert num_outputs == 1

    x, y, fov = next(iter(train_loader))
    assert x.shape[1:] == (3, 32, 40)
    assert y.shape[1:] == (1, 32, 40)
    assert fov.shape[1:] == (1, 32, 40)
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1


def test_drive_patch_training_and_pos_weight(tmp_path):
    _make_drive_tree(tmp_path / "data")
    cfg = {
        "dataset": "drive",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "drive_patch_size": 16,
        "drive_foreground_bias": 1.0,
    }

    train_loader, _, _, _, _, _ = utils.build_dataloaders(cfg)
    x, y, fov = next(iter(train_loader))
    assert x.shape[2:] == (16, 16)
    assert y.shape[2:] == (16, 16)
    assert fov.shape[2:] == (16, 16)

    pos_weight = utils.estimate_drive_pos_weight(train_loader.dataset, min_weight=1.0, max_weight=25.0)
    assert pos_weight >= 1.0


def test_segmentation_models_preserve_spatial_shape():
    x = torch.randn(2, 3, 64, 80)
    baseline = BaselineUNet(in_channels=3, out_channels=1)
    az_model = AZUNet(
        in_channels=3,
        out_channels=1,
        num_rules=3,
        cfg=AZConvConfig(geometry_mode="local_hyperbolic"),
    )

    y_base = baseline(x)
    y_az = az_model(x)
    assert y_base.shape == (2, 1, 64, 80)
    assert y_az.shape == (2, 1, 64, 80)
    assert torch.isfinite(y_az).all()


def test_az_sota_outputs_and_objective():
    x = torch.randn(2, 3, 64, 80)
    y = torch.randint(0, 2, (2, 1, 64, 80), dtype=torch.float32)
    valid = torch.ones_like(y)
    model = AZSOTAUNet(
        in_channels=3,
        out_channels=1,
        num_rules=3,
        cfg=AZConvConfig(geometry_mode="fixed_cat_map", learn_directions=False),
    )
    out = model(x)
    assert isinstance(out, dict)
    assert out["logits"].shape == (2, 1, 64, 80)
    assert len(out["aux_logits"]) == 2
    for aux in out["aux_logits"]:
        assert aux.shape == (2, 1, 64, 80)
    assert out["boundary_logits"].shape == (2, 1, 64, 80)

    loss, logs, main_logits = utils.segmentation_objective(
        out,
        y,
        valid,
        bce_weight=1.0,
        dice_weight=1.0,
        aux_weight=0.2,
        boundary_weight=0.1,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(main_logits).all()
    assert "aux_loss" in logs and logs["aux_loss"] >= 0.0
    assert "boundary_loss" in logs and logs["boundary_loss"] >= 0.0
    assert "topology_loss" in logs and logs["topology_loss"] == 0.0


def test_topology_loss_reaches_zero_for_perfect_prediction():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])
    valid = torch.ones_like(target)

    loss, logs, _ = utils.segmentation_objective(
        logits,
        target,
        valid,
        bce_weight=0.0,
        dice_weight=0.0,
        topology_weight=1.0,
        topology_num_iters=5,
    )
    assert torch.isfinite(loss)
    assert abs(logs["topology_loss"]) < 1e-4


def test_tversky_overlap_reaches_zero_for_perfect_prediction():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])
    valid = torch.ones_like(target)

    loss, logs = utils.segmentation_loss(
        logits,
        target,
        valid,
        bce_weight=0.0,
        dice_weight=1.0,
        overlap_mode="tversky",
        tversky_alpha=0.7,
        tversky_beta=0.3,
    )
    assert torch.isfinite(loss)
    assert abs(logs["dice_loss"]) < 1e-4


def test_az_sota_pure_variant_builds_and_runs():
    model = utils.build_model(
        "az_sota_pure",
        num_outputs=1,
        in_channels=3,
        num_rules=3,
        task="segmentation",
    )
    x = torch.randn(1, 3, 48, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert out["logits"].shape == (1, 1, 48, 64)
    assert len(out["aux_logits"]) == 2
    assert out["boundary_logits"].shape == (1, 1, 48, 64)


def test_az_thesis_variant_builds_and_runs():
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=3,
        task="segmentation",
    )
    x = torch.randn(1, 3, 48, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert out["logits"].shape == (1, 1, 48, 64)


def test_segmentation_metrics_reach_one_for_perfect_prediction():
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[-8.0, 8.0], [8.0, -8.0]]]])
    valid_mask = torch.ones_like(target)

    tp, fp, tn, fn = utils.binary_confusion_counts(logits, target, valid_mask, threshold=0.5)
    metrics = utils.segmentation_metrics_from_counts(tp, fp, tn, fn)

    assert metrics["dice"] > 0.999
    assert metrics["iou"] > 0.999
    assert metrics["accuracy"] > 0.999
    assert metrics["balanced_accuracy"] > 0.999


def test_threshold_grid_and_selection():
    grid = utils.build_threshold_grid(0.3, 0.5, 0.1)
    assert grid == [0.3, 0.4, 0.5]

    rows = [
        {"threshold": 0.3, "dice": 0.80},
        {"threshold": 0.4, "dice": 0.84},
        {"threshold": 0.5, "dice": 0.84},
    ]
    best = utils.select_best_threshold(rows, metric="dice", reference_threshold=0.5)
    assert best["threshold"] == 0.5


def test_drive_summary_writer_uses_best_non_smoke_run(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "az_cat_best").mkdir(parents=True)
    (results_dir / "az_cat_smoke").mkdir(parents=True)
    (results_dir / "az_thesis_run").mkdir(parents=True)

    (results_dir / "az_cat_best" / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "az_cat",
                "dataset": "drive",
                "task": "segmentation",
                "run_name": "az_cat_best",
                "seed": 42,
                "num_rules": 4,
                "aux_loss_weight": 0.2,
                "boundary_loss_weight": 0.1,
                "topology_loss_weight": 0.05,
                "selected_threshold": 0.55,
                "test_dice": 0.80,
                "test_iou": 0.67,
                "test_precision": 0.78,
                "test_recall": 0.82,
                "test_specificity": 0.96,
                "test_balanced_accuracy": 0.89,
                "seconds_per_forward_batch": 0.02,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "az_cat_smoke" / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "az_cat",
                "dataset": "drive",
                "task": "segmentation",
                "run_name": "az_cat_smoke",
                "seed": 0,
                "num_rules": 4,
                "aux_loss_weight": 0.2,
                "boundary_loss_weight": 0.1,
                "selected_threshold": 0.5,
                "test_dice": 0.99,
                "test_iou": 0.98,
                "test_precision": 0.99,
                "test_recall": 0.99,
                "test_specificity": 0.99,
                "test_balanced_accuracy": 0.99,
                "seconds_per_forward_batch": 0.01,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "az_thesis_run" / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "az_thesis",
                "dataset": "drive",
                "task": "segmentation",
                "run_name": "az_thesis_run",
                "seed": 7,
                "num_rules": 6,
                "aux_loss_weight": 0.0,
                "boundary_loss_weight": 0.1,
                "topology_loss_weight": 0.0,
                "selected_threshold": 0.6,
                "test_dice": 0.70,
                "test_iou": 0.55,
                "test_precision": 0.72,
                "test_recall": 0.69,
                "test_specificity": 0.97,
                "test_balanced_accuracy": 0.83,
                "seconds_per_forward_batch": 0.10,
            }
        ),
        encoding="utf-8",
    )

    out_path = utils.update_drive_comparison_summary(results_dir)
    content = out_path.read_text(encoding="utf-8")

    assert "az_cat_best" in content
    assert "az_cat_smoke" not in content
    assert "az_thesis_run" in content
    assert "Rules" in content
    assert "Aux w" in content
    assert "Boundary w" in content


def test_drive_superiority_gate_passes_when_candidate_beats_baseline(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "baseline_run").mkdir(parents=True)
    (results_dir / "candidate_run").mkdir(parents=True)

    (results_dir / "baseline_run" / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "baseline",
                "dataset": "drive",
                "task": "segmentation",
                "run_name": "baseline_run",
                "selected_threshold": 0.55,
                "test_dice": 0.75,
                "test_iou": 0.60,
                "test_precision": 0.76,
                "test_recall": 0.74,
                "test_specificity": 0.96,
                "test_balanced_accuracy": 0.85,
                "seconds_per_forward_batch": 0.02,
            }
        ),
        encoding="utf-8",
    )

    candidate_metrics = {
        "variant": "az_thesis",
        "run_name": "candidate_run",
        "dataset": "drive",
        "task": "segmentation",
        "test_dice": 0.76,
        "test_iou": 0.61,
        "test_precision": 0.77,
        "test_recall": 0.75,
        "test_specificity": 0.97,
        "test_balanced_accuracy": 0.86,
    }

    report = utils.compare_drive_metrics_to_baseline(results_dir, candidate_metrics)
    text = utils.format_drive_superiority_report(report)

    assert report["all_passed"] is True
    assert "PASS" in text
    assert "Dice" in text


def test_drive_superiority_gate_fails_when_any_metric_is_lower(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "baseline_run" ).mkdir(parents=True)

    (results_dir / "baseline_run" / "metrics.json").write_text(
        json.dumps(
            {
                "variant": "baseline",
                "dataset": "drive",
                "task": "segmentation",
                "run_name": "baseline_run",
                "selected_threshold": 0.55,
                "test_dice": 0.75,
                "test_iou": 0.60,
                "test_precision": 0.76,
                "test_recall": 0.74,
                "test_specificity": 0.96,
                "test_balanced_accuracy": 0.85,
                "seconds_per_forward_batch": 0.02,
            }
        ),
        encoding="utf-8",
    )

    candidate_metrics = {
        "variant": "az_cat",
        "run_name": "candidate_run",
        "dataset": "drive",
        "task": "segmentation",
        "test_dice": 0.76,
        "test_iou": 0.61,
        "test_precision": 0.77,
        "test_recall": 0.70,
        "test_specificity": 0.97,
        "test_balanced_accuracy": 0.86,
    }

    report = utils.compare_drive_metrics_to_baseline(results_dir, candidate_metrics)

    assert report["all_passed"] is False
    failed_metrics = {row["metric"] for row in report["metric_rows"] if not row["passed"]}
    assert "test_recall" in failed_metrics


def test_drive_threshold_search_report_finds_passing_threshold():
    baseline_metrics = {
        "variant": "baseline",
        "run_name": "baseline_run",
        "test_dice": 0.80,
        "test_iou": 0.67,
        "test_precision": 0.81,
        "test_recall": 0.76,
        "test_specificity": 0.97,
        "test_balanced_accuracy": 0.865,
    }
    sweep_rows = [
        {
            "threshold": 0.65,
            "dice": 0.81,
            "iou": 0.68,
            "precision": 0.79,
            "recall": 0.79,
            "specificity": 0.968,
            "balanced_accuracy": 0.879,
        },
        {
            "threshold": 0.725,
            "dice": 0.805,
            "iou": 0.675,
            "precision": 0.82,
            "recall": 0.761,
            "specificity": 0.971,
            "balanced_accuracy": 0.866,
        },
    ]

    report = utils.build_drive_threshold_search_report(
        sweep_rows=sweep_rows,
        baseline_metrics=baseline_metrics,
        selection_metric="dice",
    )
    text = utils.format_drive_threshold_search_report(report)

    assert report["all_passed"] is True
    assert report["passed_count"] == 1
    assert abs(report["best_row"]["threshold"] - 0.725) < 1e-9
    assert "PASS" in text


def test_drive_threshold_search_report_fails_when_no_threshold_dominates():
    baseline_metrics = {
        "variant": "baseline",
        "run_name": "baseline_run",
        "test_dice": 0.80,
        "test_iou": 0.67,
        "test_precision": 0.81,
        "test_recall": 0.76,
        "test_specificity": 0.97,
        "test_balanced_accuracy": 0.865,
    }
    sweep_rows = [
        {
            "threshold": 0.65,
            "dice": 0.81,
            "iou": 0.68,
            "precision": 0.79,
            "recall": 0.79,
            "specificity": 0.968,
            "balanced_accuracy": 0.879,
        },
        {
            "threshold": 0.75,
            "dice": 0.799,
            "iou": 0.668,
            "precision": 0.83,
            "recall": 0.75,
            "specificity": 0.972,
            "balanced_accuracy": 0.861,
        },
    ]

    report = utils.build_drive_threshold_search_report(
        sweep_rows=sweep_rows,
        baseline_metrics=baseline_metrics,
        selection_metric="dice",
    )

    assert report["all_passed"] is False
    assert report["passed_count"] == 0


def test_select_best_threshold_supports_core_mean_metric():
    sweep_rows = [
        {
            "threshold": 0.65,
            "dice": 0.82,
            "iou": 0.68,
            "precision": 0.79,
            "recall": 0.86,
            "specificity": 0.963,
            "balanced_accuracy": 0.912,
        },
        {
            "threshold": 0.725,
            "dice": 0.821,
            "iou": 0.69,
            "precision": 0.81,
            "recall": 0.85,
            "specificity": 0.971,
            "balanced_accuracy": 0.914,
        },
    ]

    best = utils.select_best_threshold(sweep_rows, metric="core_mean", reference_threshold=0.6)

    assert abs(best["threshold"] - 0.725) < 1e-9


def test_saved_anza_checkpoint_beats_baseline_on_drive_with_core_mean_threshold():
    if os.environ.get("RUN_DRIVE_BENCHMARK") != "1":
        pytest.skip("Set RUN_DRIVE_BENCHMARK=1 to run the saved DRIVE superiority benchmark.")

    repo_root = Path.cwd()
    baseline_metrics_path = repo_root / "results" / "multi_20260321_155448_baseline" / "metrics.json"
    candidate_checkpoint_path = repo_root / "results" / "az_cat_rules6_topology005_e120_pos45" / "checkpoint_best.pt"
    data_root = repo_root / "data" / "DRIVE"

    if not baseline_metrics_path.exists():
        pytest.skip(f"Baseline metrics are missing: {baseline_metrics_path}")
    if not candidate_checkpoint_path.exists():
        pytest.skip(f"Candidate checkpoint is missing: {candidate_checkpoint_path}")
    if not data_root.exists():
        pytest.skip(f"DRIVE data is missing: {data_root}")

    cfg = utils.load_config(str(repo_root / "configs" / "drive.yaml"))
    cfg["dataset"] = "drive"
    cfg["variant"] = "az_cat"
    cfg["num_rules"] = 6
    cfg["seed"] = 42

    utils.set_seed(int(cfg["seed"]))
    _, val_loader, test_loader, in_channels, num_outputs, task = utils.build_dataloaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = utils.build_model("az_cat", num_outputs, in_channels, num_rules=6, task=task).to(device)
    checkpoint = torch.load(candidate_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    thresholds = utils.build_threshold_grid(0.55, 0.8, 0.025)
    val_rows = utils.sweep_segmentation_thresholds(model, val_loader, device, thresholds)
    selected_row = utils.select_best_threshold(val_rows, metric="core_mean", reference_threshold=0.6)
    selected_threshold = float(selected_row["threshold"])

    test_rows = utils.sweep_segmentation_thresholds(model, test_loader, device, thresholds)
    test_row = next(row for row in test_rows if abs(float(row["threshold"]) - selected_threshold) < 1e-9)
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    candidate_metrics = {
        "variant": "az_cat",
        "run_name": "az_cat_rules6_topology005_e120_pos45@core_mean",
        "test_dice": float(test_row["dice"]),
        "test_iou": float(test_row["iou"]),
        "test_precision": float(test_row["precision"]),
        "test_recall": float(test_row["recall"]),
        "test_specificity": float(test_row["specificity"]),
        "test_balanced_accuracy": float(test_row["balanced_accuracy"]),
    }

    report = utils.build_drive_superiority_report(candidate_metrics, baseline_metrics)

    assert report["all_passed"] is True, utils.format_drive_superiority_report(report)
