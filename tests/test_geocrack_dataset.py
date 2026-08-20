from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from datasets.geocrack import GeoCrackDataset, compute_train_normalization


def _make_pair(root: Path, source: str, index: int, value: int, mask_value: int) -> dict[str, str]:
    image_path = root / "images" / f"{source}_original_patch{index}.png"
    mask_path = root / "masks" / f"{source}_binarymask_patch{index}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((224, 224, 3), value, dtype=np.uint8)).save(image_path)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[40:80, 70:90] = mask_value
    Image.fromarray(mask).save(mask_path)
    return {
        "patch_id": f"{source}_patch{index}",
        "source_image_id": source,
        "image_path": image_path.relative_to(root).as_posix(),
        "mask_path": mask_path.relative_to(root).as_posix(),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_returns_finite_binary_tensors_and_metadata(tmp_path: Path) -> None:
    rows = [_make_pair(tmp_path, "DJI_0001", 1, value=128, mask_value=255)]
    split_csv = tmp_path / "split.csv"
    _write_csv(split_csv, rows)
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}), encoding="utf-8")

    dataset = GeoCrackDataset(tmp_path, split_csv, normalization_path=normalization, augment=False)
    image, mask, metadata = dataset[0]

    assert image.shape == (3, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert image.dtype == torch.float32 and mask.dtype == torch.float32
    assert torch.isfinite(image).all() and torch.isfinite(mask).all()
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert metadata["patch_id"] == "DJI_0001_patch1"
    assert metadata["source_image_id"] == "DJI_0001"


def test_normalization_is_computed_only_from_supplied_train_csv(tmp_path: Path) -> None:
    train_rows = [
        _make_pair(tmp_path, "train_a", 1, value=0, mask_value=255),
        _make_pair(tmp_path, "train_b", 1, value=255, mask_value=255),
    ]
    _make_pair(tmp_path, "test_bright", 1, value=255, mask_value=255)
    train_csv = tmp_path / "train.csv"
    _write_csv(train_csv, train_rows)
    output = tmp_path / "train_normalization.json"

    payload = compute_train_normalization(tmp_path, train_csv, output)

    assert np.allclose(payload["mean"], [0.5, 0.5, 0.5], atol=1e-6)
    assert payload["sample_count"] == 2
    assert payload["source_csv_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
