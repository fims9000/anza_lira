#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

HF_DATASET = "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction"
HF_TREE_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main"
HF_RESOLVE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
USER_AGENT = "Mozilla/5.0"


def _request_json(url: str) -> list[dict]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def _download_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as response:
        return response.read()


def _list_files(subdir: str) -> list[str]:
    url = f"{HF_TREE_URL}/{subdir}"
    data = _request_json(url)
    return [str(item["path"]) for item in data if item.get("type") == "file"]


def _mask_from_rgb_bytes(image_bytes: bytes, threshold: int = 5) -> Image.Image:
    import io

    with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
        arr = np.asarray(image, dtype=np.uint8)
    valid = (arr.max(axis=2) > int(threshold)).astype(np.uint8) * 255
    return Image.fromarray(valid, mode="L")


def _save_bytes(path: Path, payload: bytes, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _save_mask(path: Path, image_bytes: bytes, threshold: int, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = _mask_from_rgb_bytes(image_bytes, threshold=threshold)
    mask.save(path)


def _official_names(hf_split: str, raw_name: str) -> tuple[str, str, str, str]:
    stem = Path(raw_name).stem
    sample_id = stem.split("_")[0]
    if hf_split == "train":
        local_split = "training"
        image_name = f"{sample_id}_training.tif"
        label_name = f"{sample_id}_manual1.png"
        mask_name = f"{sample_id}_training_mask.png"
    elif hf_split == "val":
        local_split = "test"
        image_name = f"{sample_id}_test.tif"
        label_name = f"{sample_id}_manual1.png"
        mask_name = f"{sample_id}_test_mask.png"
    else:
        raise ValueError(f"Unsupported split: {hf_split}")
    return local_split, image_name, label_name, mask_name


def _iter_hf_pairs(hf_split: str) -> Iterable[tuple[str, str]]:
    input_files = sorted(_list_files(f"{hf_split}/input"))
    label_files = sorted(_list_files(f"{hf_split}/label"))
    if len(input_files) != len(label_files):
        raise RuntimeError(f"Mismatched file counts for split {hf_split}: {len(input_files)} inputs vs {len(label_files)} labels")
    for input_path, label_path in zip(input_files, label_files):
        if Path(input_path).stem.split("_")[0] != Path(label_path).stem.split("_")[0]:
            raise RuntimeError(f"Input/label mismatch: {input_path} vs {label_path}")
        yield input_path, label_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Hugging Face DRIVE mirror and convert it to the repo's expected layout.")
    parser.add_argument("--data-root", type=str, default="data", help="Project data root; DRIVE will be created under <data-root>/DRIVE.")
    parser.add_argument("--threshold", type=int, default=5, help="RGB threshold used to derive a binary FOV mask from each image.")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite files if they already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    drive_root = (project_root / args.data_root / "DRIVE").resolve()

    for hf_split in ("train", "val"):
        for input_path, label_path in _iter_hf_pairs(hf_split):
            image_name = Path(input_path).name
            local_split, local_image_name, local_label_name, local_mask_name = _official_names(hf_split, image_name)
            local_base = drive_root / local_split
            image_out = local_base / "images" / local_image_name
            label_out = local_base / "1st_manual" / local_label_name
            mask_out = local_base / "mask" / local_mask_name

            image_bytes = _download_bytes(f"{HF_RESOLVE_URL}/{input_path}?download=true")
            label_bytes = _download_bytes(f"{HF_RESOLVE_URL}/{label_path}?download=true")
            _save_bytes(image_out, image_bytes, force=args.force)
            _save_bytes(label_out, label_bytes, force=args.force)
            _save_mask(mask_out, image_bytes, threshold=int(args.threshold), force=args.force)
            print(f"[{hf_split}] {image_name} -> {local_split}/{local_image_name}")

    print(f"Prepared DRIVE dataset under {drive_root}")


if __name__ == "__main__":
    main()
