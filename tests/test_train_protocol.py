from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train


_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "run_az_thesis_sweep",
    Path(__file__).resolve().parents[1] / "scripts" / "run_az_thesis_sweep.py",
)
assert _SWEEP_SPEC is not None and _SWEEP_SPEC.loader is not None
run_az_thesis_sweep = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(run_az_thesis_sweep)


class _ToySegDataset(Dataset):
    def __init__(self) -> None:
        self.foreground_bias = 0.0
        self.thin_vessel_bias = 0.0
        self.hard_mining_bias = 0.0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        x = torch.zeros(3, 8, 8)
        y = torch.zeros(1, 8, 8)
        valid = torch.ones(1, 8, 8)
        return x, y, valid

    def set_foreground_bias(self, value: float) -> None:
        self.foreground_bias = float(value)

    def set_thin_vessel_bias(self, value: float) -> None:
        self.thin_vessel_bias = float(value)

    def set_hard_mining_bias(self, value: float) -> None:
        self.hard_mining_bias = float(value)


class _TinySegModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale.view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        return {}


def _seg_metrics(dice: float) -> dict[str, float]:
    return {
        "val_loss": 1.0 - dice,
        "val_bce_loss": 0.0,
        "val_dice_loss": 1.0 - dice,
        "val_topology_loss": 0.0,
        "val_dice": dice,
        "val_iou": max(dice - 0.1, 0.0),
        "val_precision": dice,
        "val_recall": dice,
        "val_specificity": dice,
        "val_accuracy": dice,
        "val_balanced_accuracy": dice,
    }


def test_resolve_loss_cfg_uses_unbiased_pos_weight_reference(monkeypatch):
    reference = object()
    loader = SimpleNamespace(dataset=SimpleNamespace(pos_weight_reference=reference))
    called = {}

    def fake_estimate(dataset_arg, *, min_weight: float, max_weight: float) -> float:
        called["dataset"] = dataset_arg
        called["min_weight"] = min_weight
        called["max_weight"] = max_weight
        return 3.25

    monkeypatch.setattr(train, "estimate_drive_pos_weight", fake_estimate)

    loss_cfg = train.resolve_loss_cfg(
        {
            "bce_pos_weight": "auto",
            "bce_pos_weight_min": 1.0,
            "bce_pos_weight_max": 10.0,
        },
        "segmentation",
        loader,
        torch.device("cpu"),
    )

    assert called["dataset"] is reference
    assert called["min_weight"] == pytest.approx(1.0)
    assert called["max_weight"] == pytest.approx(10.0)
    assert loss_cfg["pos_weight_value"] == pytest.approx(3.25)


def test_apply_retinal_foreground_bias_schedule_linear():
    loader = DataLoader(_ToySegDataset(), batch_size=1, shuffle=False)
    cfg = {
        "retinal_foreground_bias": 0.8,
        "retinal_foreground_bias_end": 0.2,
        "retinal_foreground_bias_schedule": "linear",
    }

    first = train.apply_retinal_foreground_bias_schedule(loader, cfg, epoch=1, epochs=5)
    mid = train.apply_retinal_foreground_bias_schedule(loader, cfg, epoch=3, epochs=5)
    last = train.apply_retinal_foreground_bias_schedule(loader, cfg, epoch=5, epochs=5)

    assert first == pytest.approx(0.8)
    assert mid == pytest.approx(0.5)
    assert last == pytest.approx(0.2)


def test_apply_retinal_thin_vessel_bias_schedule_linear():
    loader = DataLoader(_ToySegDataset(), batch_size=1, shuffle=False)
    cfg = {
        "retinal_thin_vessel_bias": 0.5,
        "retinal_thin_vessel_bias_end": 0.1,
        "retinal_thin_vessel_bias_schedule": "linear",
    }

    first = train.apply_retinal_thin_vessel_bias_schedule(loader, cfg, epoch=1, epochs=5)
    mid = train.apply_retinal_thin_vessel_bias_schedule(loader, cfg, epoch=3, epochs=5)
    last = train.apply_retinal_thin_vessel_bias_schedule(loader, cfg, epoch=5, epochs=5)

    assert first == pytest.approx(0.5)
    assert mid == pytest.approx(0.3)
    assert last == pytest.approx(0.1)


def test_apply_retinal_hard_mining_bias_schedule_linear():
    loader = DataLoader(_ToySegDataset(), batch_size=1, shuffle=False)
    cfg = {
        "retinal_hard_mining_bias": 0.6,
        "retinal_hard_mining_bias_end": 0.2,
        "retinal_hard_mining_bias_schedule": "linear",
    }

    first = train.apply_retinal_hard_mining_bias_schedule(loader, cfg, epoch=1, epochs=5)
    mid = train.apply_retinal_hard_mining_bias_schedule(loader, cfg, epoch=3, epochs=5)
    last = train.apply_retinal_hard_mining_bias_schedule(loader, cfg, epoch=5, epochs=5)

    assert first == pytest.approx(0.6)
    assert mid == pytest.approx(0.4)
    assert last == pytest.approx(0.2)


def test_spatial_shape_for_run_prefers_retinal_patch_size():
    cfg = {
        "dataset": "fives",
        "retinal_patch_size": 512,
    }
    assert train.spatial_shape_for_run(cfg, "segmentation") == (512, 512)
    assert train.spatial_shape_for_run({"dataset": "fives"}, "segmentation") == (2048, 2048)


def test_run_training_selects_checkpoint_by_validation_sweep_metric(monkeypatch, tmp_path: Path):
    loader = DataLoader(_ToySegDataset(), batch_size=1, shuffle=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    epoch_state = {"train": 0, "eval": 0, "sweep": 0}

    def fake_train_one_epoch(model, *args, **kwargs):
        epoch_state["train"] += 1
        with torch.no_grad():
            model.scale.fill_(float(epoch_state["train"]))
        return {
            "train_loss": 1.0,
            "train_objective": 1.0,
        }

    eval_sequence = [
        _seg_metrics(0.90),
        _seg_metrics(0.85),
        _seg_metrics(0.82),
        _seg_metrics(0.81),
    ]

    def fake_evaluate_epoch(*args, **kwargs):
        value = eval_sequence[epoch_state["eval"]]
        epoch_state["eval"] += 1
        return value

    def fake_sweep(*args, **kwargs):
        epoch_state["sweep"] += 1
        if epoch_state["sweep"] == 1:
            return [
                {
                    "threshold": 0.55,
                    "dice": 0.70,
                    "iou": 0.70,
                    "precision": 0.70,
                    "recall": 0.70,
                    "specificity": 0.70,
                    "balanced_accuracy": 0.70,
                },
                {
                    "threshold": 0.65,
                    "dice": 0.60,
                    "iou": 0.60,
                    "precision": 0.60,
                    "recall": 0.60,
                    "specificity": 0.60,
                    "balanced_accuracy": 0.60,
                },
            ]
        return [
            {
                "threshold": 0.55,
                "dice": 0.80,
                "iou": 0.80,
                "precision": 0.80,
                "recall": 0.80,
                "specificity": 0.80,
                "balanced_accuracy": 0.80,
            },
            {
                "threshold": 0.65,
                "dice": 0.82,
                "iou": 0.82,
                "precision": 0.82,
                "recall": 0.82,
                "specificity": 0.82,
                "balanced_accuracy": 0.82,
            },
        ]

    monkeypatch.setattr(train, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "build_dataloaders", lambda cfg: (loader, loader, loader, 3, 1, "segmentation"))
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: _TinySegModel())
    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "evaluate_epoch", fake_evaluate_epoch)
    monkeypatch.setattr(train, "sweep_segmentation_thresholds", fake_sweep)
    monkeypatch.setattr(train, "measure_inference_time", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(train, "plot_single_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "update_drive_comparison_summary", lambda *args, **kwargs: run_dir / "summary.md")

    cfg = {
        "dataset": "drive",
        "seed": 0,
        "epochs": 2,
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "num_rules": 2,
        "eval_threshold_sweep": True,
        "eval_threshold_metric": "core_mean",
        "eval_threshold_start": 0.5,
        "eval_threshold_end": 0.7,
        "eval_threshold_step": 0.1,
    }

    metrics = train.run_training(cfg, variant="baseline", run_dir=run_dir)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    saved_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["best_val_selection_metric"] == "core_mean"
    assert metrics["best_val_selection_score"] == pytest.approx(0.82)
    assert "best_val_dice" not in metrics
    assert metrics["selected_threshold"] == pytest.approx(0.65)
    assert history[0]["val_selection_threshold"] == pytest.approx(0.55)
    assert history[1]["val_selection_threshold"] == pytest.approx(0.65)
    assert history[1]["val_selection_score"] == pytest.approx(0.82)
    assert saved_metrics["best_val_selection_metric"] == "core_mean"
    assert saved_metrics["selected_threshold"] == pytest.approx(0.65)
    assert (run_dir / "checkpoint_best.pt").exists()


def test_az_thesis_sweep_best_trial_artifacts_keep_loss_overrides():
    base_cfg = {
        "dataset": "drive",
        "variant": "az_thesis",
        "epochs": 30,
    }
    best_row = {
        "trial_name": "trial_demo",
        "num_rules": 8,
        "model_widths": [40, 80, 144, 208],
        "lr": 2e-4,
        "encoder_az_stages": 1,
        "encoder_block_mode": "hybrid",
        "hybrid_mix_init": 0.3,
        "boundary_loss_weight": 0.1,
        "topology_loss_weight": 0.03,
        "overlap_mode": "tversky",
        "tversky_alpha": 0.35,
        "tversky_beta": 0.65,
        "bce_pos_weight": "auto",
        "eval_threshold_metric": "core_mean",
        "bottleneck_mode": "aspp",
        "decoder_mode": "residual",
        "boundary_mode": "conv",
        "epochs": 30,
        "test_dice_mean": 0.74,
        "test_iou_mean": 0.59,
        "test_balanced_accuracy_mean": 0.85,
        "seconds_per_forward_batch_mean": 0.05,
        "seeds": [41, 42],
    }

    overrides, full_cfg, meta = run_az_thesis_sweep._best_trial_artifacts(base_cfg, best_row)

    assert overrides["overlap_mode"] == "tversky"
    assert overrides["tversky_alpha"] == pytest.approx(0.35)
    assert overrides["tversky_beta"] == pytest.approx(0.65)
    assert overrides["bce_pos_weight"] == "auto"
    assert overrides["encoder_az_stages"] == 1
    assert overrides["encoder_block_mode"] == "hybrid"
    assert overrides["hybrid_mix_init"] == pytest.approx(0.3)
    assert full_cfg["bce_pos_weight"] == "auto"
    assert full_cfg["encoder_az_stages"] == 1
    assert full_cfg["encoder_block_mode"] == "hybrid"
    assert full_cfg["hybrid_mix_init"] == pytest.approx(0.3)
    assert full_cfg["tversky_beta"] == pytest.approx(0.65)
    assert meta["trial_name"] == "trial_demo"
    assert meta["seeds"] == [41, 42]


def test_run_training_can_finetune_from_checkpoint_with_scheduler(monkeypatch, tmp_path: Path):
    loader = DataLoader(_ToySegDataset(), batch_size=1, shuffle=False)
    run_dir = tmp_path / "finetune"
    run_dir.mkdir()
    ckpt_path = tmp_path / "init.pt"
    init_model = _TinySegModel()
    with torch.no_grad():
        init_model.scale.fill_(2.0)
    torch.save({"model": init_model.state_dict(), "variant": "baseline"}, ckpt_path)

    seen = {"first_scale": None}

    def fake_train_one_epoch(model, *args, **kwargs):
        if seen["first_scale"] is None:
            seen["first_scale"] = float(model.scale.detach())
        optimizer = args[2]
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        return {
            "train_loss": 1.0,
            "train_objective": 1.0,
        }

    monkeypatch.setattr(train, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "build_dataloaders", lambda cfg: (loader, loader, loader, 3, 1, "segmentation"))
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: _TinySegModel())
    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "measure_inference_time", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(train, "plot_single_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "update_drive_comparison_summary", lambda *args, **kwargs: run_dir / "summary.md")

    cfg = {
        "dataset": "drive",
        "seed": 0,
        "epochs": 2,
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "num_rules": 2,
        "eval_threshold_sweep": False,
        "lr_scheduler": "cosine",
        "lr_min": 1e-5,
        "init_checkpoint": str(ckpt_path),
    }

    metrics = train.run_training(cfg, variant="baseline", run_dir=run_dir)
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    assert seen["first_scale"] == pytest.approx(2.0)
    assert metrics["lr_scheduler"] == "cosine"
    assert metrics["best_epoch"] == 1
    assert metrics["init_checkpoint"]["path"] == str(ckpt_path)
    assert metrics["init_checkpoint"]["variant"] == "baseline"
    assert history[0]["lr"] == pytest.approx(1e-3)
    assert history[0]["lr_next"] < history[0]["lr"]
    assert history[1]["lr"] == pytest.approx(history[0]["lr_next"])
