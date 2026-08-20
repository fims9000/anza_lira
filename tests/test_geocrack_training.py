from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from utils import build_dataloaders, build_model, spatial_shape_for_dataset, task_for_dataset, unpack_segmentation_outputs


def _write_split(root: Path, split_dir: Path, split: str, count: int) -> None:
    rows = []
    for index in range(count):
        source = f"{split}_source_{index}"
        image_path = root / "patched_data" / f"{source}_patch{index}.png"
        mask_path = root / "patched_data" / f"{source}_patch{index}_mask.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((224, 224, 3), 100 + index, dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[80:144, 108:116] = 255
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)
        rows.append(
            {
                "patch_id": f"{source}_patch{index}",
                "source_image_id": source,
                "image_path": image_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
            }
        )
    path = split_dir / f"geocrack_small_v1_{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "geocrack"
    split_dir = root / "splits"
    split_dir.mkdir(parents=True)
    for split, count in (("train", 4), ("val", 3), ("test", 2)):
        _write_split(root, split_dir, split, count)
    (split_dir / "train_normalization.json").write_text(
        json.dumps({"mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2]}), encoding="utf-8"
    )
    return root


def test_geocrack_builds_segmentation_loaders_with_valid_masks(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    cfg = {
        "dataset": "geocrack",
        "data_root": str(tmp_path),
        "batch_size": 2,
        "num_workers": 0,
        "seed": 42,
        "geocrack_train_limit": 3,
        "geocrack_val_limit": 2,
        "geocrack_test_limit": 1,
        "geocrack_augment": False,
    }
    train_loader, val_loader, test_loader, in_channels, outputs, task = build_dataloaders(cfg)
    image, mask, valid_mask = next(iter(train_loader))

    assert (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)) == (3, 2, 1)
    assert image.shape[1:] == (3, 224, 224)
    assert mask.shape[1:] == valid_mask.shape[1:] == (1, 224, 224)
    assert torch.all(valid_mask == 1)
    assert (in_channels, outputs, task) == (3, 1, "segmentation")
    assert task_for_dataset("geocrack") == "segmentation"
    assert spatial_shape_for_dataset("geocrack") == (224, 224)


def test_existing_model_factory_builds_required_geocrack_variants() -> None:
    image = torch.randn(1, 3, 64, 64)
    for variant in ("baseline", "attention_unet", "az_no_fuzzy", "az_no_aniso", "az_thesis"):
        model = build_model(
            variant,
            num_outputs=1,
            in_channels=3,
            task="segmentation",
            widths=(4, 8, 12, 16),
            num_rules=2,
        )
        output = model(image)
        logits, _, _ = unpack_segmentation_outputs(output)
        assert logits.shape == (1, 1, 64, 64)
        assert torch.isfinite(logits).all()
