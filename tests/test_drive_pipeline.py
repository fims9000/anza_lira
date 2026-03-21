from __future__ import annotations

import numpy as np
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
