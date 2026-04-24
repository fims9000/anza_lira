from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import utils
from models import AZSOTAUNet, AZUNet, AttentionUNet, BaselineUNet
from models.azconv import AZConvConfig
from models.segmentation import AZResidualBlock, HybridAZResidualBlock, ResidualConvBlock


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


def _make_chase_tree(root) -> None:
    samples = [
        ("training", "Image_01L"),
        ("training", "Image_01R"),
        ("test", "Image_02L"),
    ]
    for split, sample_id in samples:
        split_dir = root / "CHASE_DB1" / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "1st_manual").mkdir(parents=True, exist_ok=True)
        (split_dir / "mask").mkdir(parents=True, exist_ok=True)

        _write_rgb(split_dir / "images" / f"{sample_id}.jpg")
        _write_mask(split_dir / "1st_manual" / f"{sample_id}_manual1.png")
        _write_mask(split_dir / "mask" / f"{sample_id}_mask.png", invert=True)


def _make_fives_tree(root) -> None:
    samples = [
        ("training", "FIVES_0001"),
        ("training", "FIVES_0002"),
        ("test", "FIVES_0201"),
    ]
    for split, sample_id in samples:
        split_dir = root / "FIVES" / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "1st_manual").mkdir(parents=True, exist_ok=True)
        (split_dir / "mask").mkdir(parents=True, exist_ok=True)

        _write_rgb(split_dir / "images" / f"{sample_id}.png", height=48, width=48)
        _write_mask(split_dir / "1st_manual" / f"{sample_id}_manual1.png", height=48, width=48)
        _write_mask(split_dir / "mask" / f"{sample_id}_mask.png", height=48, width=48, invert=True)


def _make_arcade_tree(root) -> None:
    for split, sample_ids in (
        ("train", [1, 2]),
        ("val", [3]),
        ("test", [4]),
    ):
        split_dir = root / "ARCADE" / "arcade" / "syntax" / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "annotations").mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1
        for sample_id in sample_ids:
            file_name = f"{sample_id}.png"
            _write_mask(split_dir / "images" / file_name, height=64, width=64)
            images.append({"id": sample_id, "file_name": file_name, "height": 64, "width": 64})
            polygon = [16, 16, 48, 16, 48, 48, 16, 48]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": sample_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [16, 16, 32, 32],
                    "area": 1024,
                    "segmentation": [polygon],
                    "attributes": {},
                }
            )
            ann_id += 1

        payload = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "vessel", "supercategory": ""}],
        }
        with open(split_dir / "annotations" / f"{split}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


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


def test_chase_db1_dataloaders_build_and_batch(tmp_path):
    _make_chase_tree(tmp_path / "data")
    cfg = {
        "dataset": "chase_db1",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "retinal_patch_size": 16,
        "retinal_foreground_bias": 1.0,
    }

    train_loader, val_loader, test_loader, in_channels, num_outputs, task = utils.build_dataloaders(cfg)
    assert task == "segmentation"
    assert in_channels == 3
    assert num_outputs == 1

    x, y, fov = next(iter(train_loader))
    assert x.shape[2:] == (16, 16)
    assert y.shape[1:] == (1, 16, 16)
    assert fov.shape[1:] == (1, 16, 16)
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1


def test_fives_dataloaders_build_and_batch(tmp_path):
    _make_fives_tree(tmp_path / "data")
    cfg = {
        "dataset": "fives",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "retinal_patch_size": 24,
        "retinal_foreground_bias": 1.0,
    }

    train_loader, val_loader, test_loader, in_channels, num_outputs, task = utils.build_dataloaders(cfg)
    assert task == "segmentation"
    assert in_channels == 3
    assert num_outputs == 1

    x, y, fov = next(iter(train_loader))
    assert x.shape[2:] == (24, 24)
    assert y.shape[1:] == (1, 24, 24)
    assert fov.shape[1:] == (1, 24, 24)
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1


def test_arcade_syntax_dataloaders_build_and_batch(tmp_path):
    _make_arcade_tree(tmp_path / "data")
    cfg = {
        "dataset": "arcade_syntax",
        "data_root": str(tmp_path / "data"),
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "arcade_patch_size": 32,
        "arcade_train_limit": 1,
        "arcade_val_limit": 1,
        "arcade_test_limit": 1,
    }

    train_loader, val_loader, test_loader, in_channels, num_outputs, task = utils.build_dataloaders(cfg)
    assert task == "segmentation"
    assert in_channels == 3
    assert num_outputs == 1

    x, y, valid = next(iter(train_loader))
    assert x.shape[1:] == (3, 32, 32)
    assert y.shape[1:] == (1, 32, 32)
    assert valid.shape[1:] == (1, 32, 32)
    assert float(y.sum()) > 0.0
    assert len(val_loader.dataset) == 1
    assert len(test_loader.dataset) == 1


def test_retinal_photometric_augmentation_changes_image_but_preserves_masks(tmp_path, monkeypatch):
    _make_chase_tree(tmp_path / "data")
    dataset = utils.DriveDataset(
        tmp_path / "data" / "CHASE_DB1",
        split="training",
        augment=True,
        use_fov_mask=True,
        brightness_jitter=0.2,
        contrast_jitter=0.2,
        gamma_jitter=0.2,
    )

    image_path, mask_path, fov_path = dataset.samples[0]
    image = dataset._load_rgb(image_path)
    mask = dataset._load_mask(mask_path)
    fov = dataset._load_mask(fov_path)

    monkeypatch.setattr(random, "random", lambda: 1.0)
    jitter_values = iter([1.1, 1.15, 0.9])
    monkeypatch.setattr(random, "uniform", lambda a, b: next(jitter_values))

    aug_image, aug_mask, aug_fov = dataset._apply_augment(image.clone(), mask.clone(), fov.clone())

    assert not torch.allclose(aug_image, image)
    assert torch.equal(aug_mask, mask)
    assert torch.equal(aug_fov, fov)
    assert float(aug_image.min()) >= 0.0
    assert float(aug_image.max()) <= 1.0


def test_drive_dataloader_plumbs_retinal_photometric_config(tmp_path):
    _make_chase_tree(tmp_path / "data")
    cfg = {
        "dataset": "chase_db1",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "retinal_brightness_jitter": 0.12,
        "retinal_contrast_jitter": 0.08,
        "retinal_gamma_jitter": 0.1,
    }

    train_loader, _, _, _, _, _ = utils.build_dataloaders(cfg)
    train_dataset = train_loader.dataset.dataset

    assert train_dataset.brightness_jitter == pytest.approx(0.12)
    assert train_dataset.contrast_jitter == pytest.approx(0.08)
    assert train_dataset.gamma_jitter == pytest.approx(0.1)


def test_drive_dataloader_plumbs_thin_vessel_sampling_config(tmp_path):
    _make_chase_tree(tmp_path / "data")
    cfg = {
        "dataset": "chase_db1",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "retinal_thin_vessel_bias": 0.35,
        "retinal_thin_vessel_neighbor_threshold": 3,
    }

    train_loader, _, _, _, _, _ = utils.build_dataloaders(cfg)
    train_dataset = train_loader.dataset.dataset

    assert train_dataset.thin_vessel_bias == pytest.approx(0.35)
    assert train_dataset.thin_vessel_neighbor_threshold == 3


def test_drive_dataloader_plumbs_hard_mining_config(tmp_path):
    _make_chase_tree(tmp_path / "data")
    mining_root = tmp_path / "mining"
    mining_root.mkdir(parents=True, exist_ok=True)
    _write_mask(mining_root / "Image_01L.png", height=32, width=40)
    cfg = {
        "dataset": "chase_db1",
        "data_root": str(tmp_path / "data"),
        "val_fraction": 0.5,
        "batch_size": 1,
        "num_workers": 0,
        "seed": 0,
        "use_fov_mask": True,
        "retinal_hard_mining_dir": str(mining_root),
        "retinal_hard_mining_bias": 0.4,
    }

    train_loader, _, _, _, _, _ = utils.build_dataloaders(cfg)
    train_dataset = train_loader.dataset.dataset

    assert train_dataset.hard_mining_dir == mining_root
    assert train_dataset.hard_mining_bias == pytest.approx(0.4)


def test_retinal_input_mode_green_equalized_replicates_green_channel(tmp_path):
    _make_chase_tree(tmp_path / "data")
    dataset = utils.DriveDataset(
        tmp_path / "data" / "CHASE_DB1",
        split="training",
        augment=False,
        use_fov_mask=True,
        input_mode="green_equalized",
    )

    image_path, _, _ = dataset.samples[0]
    image = dataset._load_rgb(image_path)
    assert image.shape[0] == 3
    assert torch.allclose(image[0], image[1])
    assert torch.allclose(image[1], image[2])


def test_retinal_input_mode_green_hybrid_preserves_rgb_difference(tmp_path):
    _make_chase_tree(tmp_path / "data")
    dataset = utils.DriveDataset(
        tmp_path / "data" / "CHASE_DB1",
        split="training",
        augment=False,
        use_fov_mask=True,
        input_mode="green_hybrid",
        green_blend_alpha=0.4,
    )

    image_path, _, _ = dataset.samples[0]
    image = dataset._load_rgb(image_path)
    assert image.shape[0] == 3
    assert not torch.allclose(image[0], image[1])
    assert not torch.allclose(image[1], image[2])


def test_thin_vessel_candidates_focus_on_sparse_vessel_pixels(tmp_path):
    _make_chase_tree(tmp_path / "data")
    dataset = utils.DriveDataset(
        tmp_path / "data" / "CHASE_DB1",
        split="training",
        augment=False,
        use_fov_mask=True,
        thin_vessel_neighbor_threshold=3,
    )

    mask = torch.zeros(1, 9, 9)
    mask[:, 1:4, 1:4] = 1.0
    mask[:, 6, 1:8] = 1.0
    fov = torch.ones_like(mask)

    thin = dataset._thin_vessel_candidates(mask, fov)

    assert thin[0, 6, 1]
    assert thin[0, 6, 4]
    assert not thin[0, 2, 2]


def test_sample_weighted_anchor_picks_hotspot():
    weights = torch.zeros(1, 8, 10)
    weights[0, 6, 8] = 5.0
    top, left = utils.DriveDataset._sample_weighted_anchor(weights, crop_h=4, crop_w=4, max_top=4, max_left=6)
    assert (top, left) == (4, 6)


def test_segmentation_tta_wrapper_flips_averages_logits():
    class _CoordLogitModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            width = x.shape[-1]
            grid = torch.linspace(0.0, 1.0, width, dtype=x.dtype, device=x.device)
            return grid.view(1, 1, 1, width).expand(x.shape[0], 1, x.shape[-2], width)

    x = torch.zeros(2, 3, 8, 10)
    wrapped = utils.SegmentationTTAWrapper(_CoordLogitModel(), mode="flips")
    out = wrapped(x)

    assert isinstance(out, dict)
    logits = out["logits"]
    assert logits.shape == (2, 1, 8, 10)
    assert torch.allclose(logits, torch.full_like(logits, 0.5), atol=1e-6)


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


def test_drive_validation_protocol_stays_full_image_and_unbiased(tmp_path):
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

    train_loader, val_loader, _, _, _, _ = utils.build_dataloaders(cfg)
    train_x, _, _ = next(iter(train_loader))
    val_x, _, _ = next(iter(val_loader))

    assert train_x.shape[2:] == (16, 16)
    assert val_x.shape[2:] == (32, 40)
    assert hasattr(train_loader.dataset, "pos_weight_reference")

    ref_x, _, _ = train_loader.dataset.pos_weight_reference[0]
    assert ref_x.shape[1:] == (32, 40)

    biased_weight = utils.estimate_drive_pos_weight(train_loader.dataset, min_weight=1.0, max_weight=25.0)
    unbiased_weight = utils.estimate_drive_pos_weight(
        train_loader.dataset.pos_weight_reference,
        min_weight=1.0,
        max_weight=25.0,
    )

    assert unbiased_weight > biased_weight


def test_segmentation_models_preserve_spatial_shape():
    x = torch.randn(2, 3, 64, 80)
    baseline = BaselineUNet(in_channels=3, out_channels=1)
    attention = AttentionUNet(in_channels=3, out_channels=1)
    az_model = AZUNet(
        in_channels=3,
        out_channels=1,
        num_rules=3,
        cfg=AZConvConfig(geometry_mode="local_hyperbolic"),
    )

    y_base = baseline(x)
    y_attention = attention(x)
    y_az = az_model(x)
    assert y_base.shape == (2, 1, 64, 80)
    assert y_attention.shape == (2, 1, 64, 80)
    assert y_az.shape == (2, 1, 64, 80)
    assert torch.isfinite(y_attention).all()
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


def test_az_thesis_hybrid_modes_build_and_run():
    cfg = {
        "encoder_az_stages": 1,
        "encoder_block_mode": "hybrid",
        "hybrid_mix_init": 0.3,
        "bottleneck_mode": "az_single",
        "decoder_mode": "residual",
        "boundary_mode": "conv",
    }
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=4,
        task="segmentation",
        widths=(32, 64, 128, 192),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
    )

    x = torch.randn(2, 3, 64, 80)
    out = model(x)

    assert isinstance(model, AZSOTAUNet)
    assert model.encoder_az_stages == 1
    assert model.encoder_block_mode == "hybrid"
    assert model.hybrid_mix_init == pytest.approx(0.3)
    assert isinstance(model.enc1, HybridAZResidualBlock)
    assert isinstance(model.enc2, ResidualConvBlock)
    assert isinstance(model.enc3, ResidualConvBlock)
    assert torch.sigmoid(model.enc1.mix_logit).item() == pytest.approx(0.3, abs=1e-4)
    assert model.bottleneck_mode == "az_single"
    assert model.decoder_mode == "residual"
    assert model.boundary_mode == "conv"
    assert out["logits"].shape == (2, 1, 64, 80)
    assert len(out["aux_logits"]) == 2
    assert out["boundary_logits"].shape == (2, 1, 64, 80)
    assert torch.isfinite(out["logits"]).all()


def test_az_thesis_hybrid_shallow_mode_builds_and_runs():
    cfg = {
        "encoder_az_stages": 2,
        "encoder_block_mode": "hybrid_shallow",
        "hybrid_mix_init": 0.4,
        "bottleneck_mode": "az_single",
        "decoder_mode": "residual",
        "boundary_mode": "conv",
    }
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=4,
        task="segmentation",
        widths=(32, 64, 128, 192),
        model_kwargs=utils.resolve_segmentation_model_kwargs(cfg),
    )

    x = torch.randn(1, 3, 64, 80)
    out = model(x)

    assert isinstance(model.enc1, HybridAZResidualBlock)
    assert isinstance(model.enc2, AZResidualBlock)
    assert isinstance(model.enc3, ResidualConvBlock)
    assert model.encoder_block_mode == "hybrid_shallow"
    assert torch.sigmoid(model.enc1.mix_logit).item() == pytest.approx(0.4, abs=1e-4)
    assert out["logits"].shape == (1, 1, 64, 80)


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


def test_attention_unet_variant_builds_and_runs():
    model = utils.build_model(
        "attention_unet",
        num_outputs=1,
        in_channels=3,
        task="segmentation",
    )
    x = torch.randn(1, 3, 48, 64)
    out = model(x)
    assert out.shape == (1, 1, 48, 64)


def test_segmentation_build_model_accepts_custom_widths():
    model = utils.build_model(
        "az_thesis",
        num_outputs=1,
        in_channels=3,
        num_rules=3,
        task="segmentation",
        widths=(24, 48, 72, 96),
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out["logits"].shape == (1, 1, 64, 64)


def test_model_complexity_summary_reports_positive_macs():
    model = utils.build_model(
        "attention_unet",
        num_outputs=1,
        in_channels=3,
        task="segmentation",
        widths=(16, 32, 48, 64),
    )
    summary = utils.estimate_model_complexity(
        model,
        device=torch.device("cpu"),
        batch_size=1,
        in_channels=3,
        spatial_shape=(64, 64),
    )
    assert summary["approx_macs_per_forward"] > 0.0
    assert summary["approx_flops_per_forward"] == pytest.approx(summary["approx_macs_per_forward"] * 2.0)


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


def test_drive_multiseed_summary_aggregates_mean_std_and_delta(tmp_path):
    results_dir = tmp_path / "results"
    for run_name, variant, seed, dice, threshold in [
        ("baseline_seed_41", "baseline", 41, 0.7800, 0.70),
        ("baseline_seed_42", "baseline", 42, 0.8000, 0.75),
        ("az_cat_seed_41", "az_cat", 41, 0.7900, 0.65),
        ("az_cat_seed_42", "az_cat", 42, 0.8100, 0.70),
    ]:
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "dataset": "drive",
                    "task": "segmentation",
                    "run_name": run_name,
                    "seed": seed,
                    "num_rules": 6 if variant != "baseline" else None,
                    "topology_loss_weight": 0.03 if variant != "baseline" else 0.0,
                    "selected_threshold": threshold,
                    "test_dice": dice,
                    "test_iou": dice - 0.12,
                    "test_precision": dice + 0.01,
                    "test_recall": dice - 0.01,
                    "test_specificity": 0.97,
                    "test_balanced_accuracy": dice - 0.02,
                    "seconds_per_forward_batch": 0.02 if variant == "baseline" else 0.05,
                }
            ),
            encoding="utf-8",
        )

    out_path = utils.update_drive_multiseed_summary(results_dir, variants=["baseline", "az_cat"])
    content = out_path.read_text(encoding="utf-8")

    assert "baseline_seed_42" in content
    assert "az_cat_seed_42" in content
    assert "0.7900 +- 0.0100" in content
    assert "0.8000 +- 0.0100" in content
    assert "+0.0100" in content


def test_segmentation_multiseed_summary_supports_non_drive_dataset(tmp_path):
    results_dir = tmp_path / "results"
    for run_name, variant, seed, dice, threshold in [
        ("baseline_seed_41", "baseline", 41, 0.7000, 0.70),
        ("baseline_seed_42", "baseline", 42, 0.7200, 0.75),
        ("az_thesis_seed_41", "az_thesis", 41, 0.7300, 0.70),
        ("az_thesis_seed_42", "az_thesis", 42, 0.7400, 0.75),
    ]:
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "dataset": "chase_db1",
                    "task": "segmentation",
                    "run_name": run_name,
                    "seed": seed,
                    "num_rules": 8 if variant != "baseline" else None,
                    "selected_threshold": threshold,
                    "test_dice": dice,
                    "test_iou": dice - 0.12,
                    "test_precision": dice + 0.01,
                    "test_recall": dice - 0.01,
                    "test_specificity": 0.97,
                    "test_balanced_accuracy": dice - 0.02,
                    "seconds_per_forward_batch": 0.02 if variant == "baseline" else 0.05,
                }
            ),
            encoding="utf-8",
        )

    out_path = utils.update_segmentation_multiseed_summary(
        results_dir,
        dataset="chase_db1",
        variants=["baseline", "az_thesis"],
    )
    content = out_path.read_text(encoding="utf-8")

    assert out_path.name == "chase_db1_multiseed_summary.md"
    assert "baseline_seed_42" in content
    assert "az_thesis_seed_42" in content
    assert "0.7350 +- 0.0050" in content
    assert "+0.0250" in content


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


def test_az_regularization_weights_include_anisotropy_gap():
    weights = utils.az_regularization_weights({"reg_anisotropy_gap": 0.25})

    assert weights["anisotropy_gap"] == pytest.approx(0.25)


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
