#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


def _iter_images(raw_root: Path) -> list[Path]:
    supervisely_img_root = raw_root / "ds0" / "img"
    if supervisely_img_root.exists():
        return sorted(supervisely_img_root.glob("*.*"))
    candidates: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        candidates.extend(raw_root.glob(pattern))
    images = []
    for path in candidates:
        stem = path.stem
        if stem.endswith("_1stHO") or stem.endswith("_2ndHO") or stem.endswith("_mask"):
            continue
        images.append(path)
    return sorted(set(images))


def _find_label(raw_root: Path, sample_id: str) -> Path:
    supervisely_ann = raw_root / "ds0" / "ann" / f"{sample_id}.jpg.json"
    if supervisely_ann.exists():
        return supervisely_ann
    for suffix in ("_1stHO.png", "_1stHO.gif", "_1stHO.tif", "_1stHO.tiff"):
        candidate = raw_root / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing first annotator label for {sample_id} under {raw_root}")


def _save_mask_from_image(image_path: Path, out_path: Path, threshold: int, force: bool) -> None:
    if out_path.exists() and not force:
        return
    with Image.open(image_path).convert("RGB") as image:
        arr = np.asarray(image, dtype=np.uint8)
    valid = (arr.max(axis=2) > int(threshold)).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(valid, mode="L").save(out_path)


def _copy_image(image_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_path.read_bytes())


def _copy_label(label_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if label_path.suffix.lower() == ".json":
        data = json.loads(label_path.read_text(encoding="utf-8"))
        height = int(data["size"]["height"])
        width = int(data["size"]["width"])
        mask = np.zeros((height, width), dtype=np.uint8)
        for obj in data.get("objects", []):
            bitmap = obj.get("bitmap")
            if not bitmap:
                continue
            packed = base64.b64decode(bitmap["data"])
            png_bytes = zlib.decompress(packed)
            with Image.open(io.BytesIO(png_bytes)).convert("RGBA") as image:
                rgba = np.asarray(image, dtype=np.uint8)
            alpha = rgba[..., 3] > 0
            origin_x, origin_y = bitmap.get("origin", [0, 0])
            y0 = int(origin_y)
            x0 = int(origin_x)
            y1 = min(y0 + alpha.shape[0], height)
            x1 = min(x0 + alpha.shape[1], width)
            cropped = alpha[: max(y1 - y0, 0), : max(x1 - x0, 0)]
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], cropped.astype(np.uint8) * 255)
        Image.fromarray(mask, mode="L").save(out_path)
        return
    with Image.open(label_path).convert("L") as label:
        arr = (np.asarray(label, dtype=np.uint8) > 127).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CHASE_DB1 into the repo's DRIVE-style retinal vessel layout.",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/CHASE_DB1_raw",
        help="Directory containing raw CHASE_DB1 images and *_1stHO annotations.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Project data root; normalized CHASE_DB1 will be created under <data-root>/CHASE_DB1.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=8,
        help="How many sorted images to place into the training split. Remaining images go to test.",
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
    out_root = (project_root / args.data_root / "CHASE_DB1").resolve()
    if not raw_root.exists():
        raise SystemExit(f"Raw CHASE_DB1 directory was not found: {raw_root}")

    image_paths = _iter_images(raw_root)
    if not image_paths:
        raise SystemExit(f"No raw CHASE_DB1 image files were found under {raw_root}")
    if int(args.train_count) <= 0 or int(args.train_count) >= len(image_paths):
        raise SystemExit(f"--train-count must be between 1 and {len(image_paths) - 1}, got {args.train_count}")

    train_ids = {path.stem for path in image_paths[: int(args.train_count)]}
    for image_path in image_paths:
        sample_id = image_path.stem
        split = "training" if sample_id in train_ids else "test"
        label_path = _find_label(raw_root, sample_id)
        split_root = out_root / split

        image_out = split_root / "images" / image_path.name
        label_out = split_root / "1st_manual" / f"{sample_id}_manual1.png"
        mask_out = split_root / "mask" / f"{sample_id}_mask.png"

        _copy_image(image_path, image_out, force=args.force)
        _copy_label(label_path, label_out, force=args.force)
        _save_mask_from_image(image_path, mask_out, threshold=int(args.threshold), force=args.force)
        print(f"[{split}] {sample_id}")

    print(f"Prepared CHASE_DB1 dataset under {out_root}")


if __name__ == "__main__":
    main()
