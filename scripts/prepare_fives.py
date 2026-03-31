#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
IMAGE_DIR_CANDIDATES = ("Original", "original", "images", "Images", "image", "Image")
MASK_DIR_CANDIDATES = (
    "Ground Truth",
    "GroundTruth",
    "ground_truth",
    "groundtruth",
    "Manual",
    "manual",
    "mask",
    "Mask",
    "1st_manual",
)
SPLIT_CANDIDATES = {
    "training": ("train", "training", "Train", "Training"),
    "test": ("test", "testing", "Test", "Testing"),
}


def _find_existing_subdir(root: Path, candidates: tuple[str, ...] | list[str]) -> Path:
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the expected directories exist under {root}: {', '.join(candidates)}")


def _iter_image_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        paths.extend(root.glob(pattern))
    return sorted(set(path for path in paths if path.is_file()))


def _sample_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = stem.replace(" ", "_").replace("-", "_")
    for suffix in (
        "_manual1",
        "_manual",
        "_mask",
        "_ground_truth",
        "_groundtruth",
        "_gt",
        "_vessel",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"__+", "_", stem)
    return stem


def _copy_image(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _copy_binary_mask(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src).convert("L") as image:
        arr = (np.asarray(image, dtype=np.uint8) > 127).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(dst)


def _derive_fov_mask(image_path: Path, out_path: Path, threshold: int, force: bool) -> None:
    if out_path.exists() and not force:
        return
    with Image.open(image_path).convert("RGB") as image:
        arr = np.asarray(image, dtype=np.uint8)
    valid = (arr.max(axis=2) > int(threshold)).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(valid, mode="L").save(out_path)


def _prepare_split(raw_root: Path, out_root: Path, split_name: str, threshold: int, force: bool) -> int:
    split_root = _find_existing_subdir(raw_root, SPLIT_CANDIDATES[split_name])
    image_root = _find_existing_subdir(split_root, IMAGE_DIR_CANDIDATES)
    mask_root = _find_existing_subdir(split_root, MASK_DIR_CANDIDATES)

    image_map = {_sample_key(path): path for path in _iter_image_files(image_root)}
    mask_map = {_sample_key(path): path for path in _iter_image_files(mask_root)}
    common = sorted(set(image_map) & set(mask_map))
    if not common:
        raise FileNotFoundError(
            f"No matched image/mask pairs found for split '{split_name}' under {split_root}"
        )

    count = 0
    for key in common:
        image_path = image_map[key]
        mask_path = mask_map[key]
        sample_id = image_path.stem
        split_out = out_root / split_name
        image_out = split_out / "images" / image_path.name
        mask_out = split_out / "1st_manual" / f"{sample_id}_manual1.png"
        fov_out = split_out / "mask" / f"{sample_id}_mask.png"

        _copy_image(image_path, image_out, force=force)
        _copy_binary_mask(mask_path, mask_out, force=force)
        _derive_fov_mask(image_path, fov_out, threshold=threshold, force=force)
        print(f"[{split_name}] {sample_id}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FIVES into the repo's normalized retinal vessel layout.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/FIVES_raw",
        help="Directory containing raw FIVES train/test folders.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Project data root; normalized FIVES will be created under <data-root>/FIVES.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="RGB threshold used to derive a binary FOV mask from each image.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite files if they already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_root = (project_root / args.source_root).resolve()
    out_root = (project_root / args.data_root / "FIVES").resolve()
    if not raw_root.exists():
        raise SystemExit(f"Raw FIVES directory was not found: {raw_root}")

    train_count = _prepare_split(raw_root, out_root, "training", threshold=int(args.threshold), force=args.force)
    test_count = _prepare_split(raw_root, out_root, "test", threshold=int(args.threshold), force=args.force)
    print(f"Prepared FIVES dataset under {out_root} ({train_count} training, {test_count} test)")


if __name__ == "__main__":
    main()
