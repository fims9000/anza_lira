#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image


ZENODO_RECORD_URL = "https://zenodo.org/records/16744782"
ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/16744782/files/HRF-Seg%2B.zip?download=1"


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(out_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)


def _find_dataset_root(extract_dir: Path) -> Path:
    for candidate in [extract_dir / "HRF-Seg+", *extract_dir.rglob("HRF-Seg+")]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find HRF-Seg+ folder under {extract_dir}")


def _is_training_sample(sample_id: str, train_index_count: int) -> bool:
    prefix = sample_id.split("_", 1)[0]
    try:
        return int(prefix) <= int(train_index_count)
    except ValueError:
        return True


def _write_binary_mask_from_rgba(mask_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    with Image.open(mask_path).convert("RGBA") as image:
        arr = np.asarray(image, dtype=np.uint8)
    alpha = arr[..., 3] > 0
    if alpha.any() and not alpha.all():
        foreground = alpha
    else:
        # Fallback for flattened RGB masks without transparency.
        foreground = np.abs(arr[..., :3].astype(np.int16) - 255).sum(axis=2) > 20
    mask = np.where(foreground, 255, 0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)


def _write_full_fov(image_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    with Image.open(image_path) as image:
        width, height = image.size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (width, height), 255).save(out_path)


def _copy_rgb(image_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert("RGB") as image:
        image.save(out_path)


def _prepare(extracted_root: Path, out_root: Path, train_index_count: int, force: bool) -> tuple[int, int]:
    images_dir = extracted_root / "Folder_6_Original_Images"
    vessels_dir = extracted_root / "Folder_3_Vessels"
    if not images_dir.exists() or not vessels_dir.exists():
        raise FileNotFoundError(
            f"Expected Folder_6_Original_Images and Folder_3_Vessels under {extracted_root}"
        )

    image_paths = sorted([p for p in images_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No HRF-Seg+ original images found under {images_dir}")

    train_count = 0
    test_count = 0
    for image_path in image_paths:
        sample_id = image_path.stem
        vessel_path = vessels_dir / f"{sample_id}.png"
        if not vessel_path.exists():
            raise FileNotFoundError(f"Missing vessel mask for {sample_id}: {vessel_path}")
        split = "training" if _is_training_sample(sample_id, train_index_count) else "test"
        split_root = out_root / split
        image_out = split_root / "images" / f"{sample_id}.png"
        manual_out = split_root / "1st_manual" / f"{sample_id}_manual1.png"
        fov_out = split_root / "mask" / f"{sample_id}_mask.png"

        _copy_rgb(image_path, image_out, force=force)
        _write_binary_mask_from_rgba(vessel_path, manual_out, force=force)
        _write_full_fov(image_path, fov_out, force=force)
        if split == "training":
            train_count += 1
        else:
            test_count += 1
        print(f"[{split}] {sample_id}")

    return train_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HRF-Seg+ from Zenodo and prepare a DRIVE-style retinal vessel dataset.",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Project data root.")
    parser.add_argument("--force", action="store_true", help="Redownload/re-extract/overwrite prepared files.")
    parser.add_argument(
        "--train-index-count",
        type=int,
        default=10,
        help="Use numeric image groups 1..N for training; remaining groups become test.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (project_root / args.data_root).resolve()
    zip_path = data_root / "_downloads" / "hrf_seg_plus.zip"
    raw_root = data_root / "HRF_SegPlus_raw"
    extract_dir = raw_root / "extracted"
    out_root = data_root / "HRF_SegPlus"

    if args.force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    if args.force or not zip_path.exists():
        print(f"Downloading HRF-Seg+ from {ZENODO_RECORD_URL} -> {zip_path}")
        _download(ZENODO_DOWNLOAD_URL, zip_path)
    else:
        print(f"Reusing existing archive: {zip_path}")

    if args.force or not extract_dir.exists():
        print(f"Extracting -> {extract_dir}")
        _extract(zip_path, extract_dir)
    else:
        print(f"Reusing extracted files: {extract_dir}")

    dataset_root = _find_dataset_root(extract_dir)
    train_count, test_count = _prepare(dataset_root, out_root, train_index_count=args.train_index_count, force=args.force)
    print(f"Prepared HRF_SegPlus under {out_root}")
    print(f"Samples: training={train_count}, test={test_count}")
    print("Source: HRF-Seg+, Zenodo record 10.5281/zenodo.16744782, CC BY 4.0")


if __name__ == "__main__":
    main()
