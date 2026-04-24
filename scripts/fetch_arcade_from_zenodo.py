#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def _download_file(url: str, out_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "anza-lira/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", "0"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as handle:
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(f"\rDownloading arcade.zip: {pct:5.1f}% ({downloaded // (1024 * 1024)} MB)", end="", flush=True)
        if total > 0:
            print()


def _extract_zip(zip_path: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def _resolve_download_url(record_id: str, file_name: str) -> str:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        payload = json.load(response)
    files = payload.get("files", [])
    for item in files:
        if item.get("key") == file_name:
            return str(item["links"]["self"])
    raise RuntimeError(f"File '{file_name}' was not found in Zenodo record {record_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract ARCADE coronary angiography dataset from Zenodo.")
    parser.add_argument("--record-id", type=str, default="10390295")
    parser.add_argument("--file-name", type=str, default="arcade.zip")
    parser.add_argument("--zip-path", type=str, default="data/arcade.zip")
    parser.add_argument("--extract-dir", type=str, default="data/ARCADE")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract archive even if target directory exists.")
    parser.add_argument("--force-download", action="store_true", help="Re-download archive even if zip file exists.")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).resolve()
    extract_dir = Path(args.extract_dir).resolve()

    if not zip_path.exists() or args.force_download:
        print(f"Resolving Zenodo URL for record {args.record_id} ...")
        download_url = _resolve_download_url(args.record_id, args.file_name)
        tmp_zip = Path(tempfile.gettempdir()) / f"{args.file_name}.part"
        if tmp_zip.exists():
            tmp_zip.unlink()
        _download_file(download_url, tmp_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_zip), str(zip_path))
        print(f"Saved archive to {zip_path}")
    else:
        print(f"Archive already exists: {zip_path}")

    needs_extract = args.force_extract or not (extract_dir / "arcade").exists()
    if needs_extract:
        print(f"Extracting {zip_path} -> {extract_dir}")
        _extract_zip(zip_path, extract_dir, force=args.force_extract)
    else:
        print(f"Extraction target already exists: {extract_dir / 'arcade'}")

    print("ARCADE dataset is ready.")
    print("Expected split layout:")
    print("  data/ARCADE/arcade/syntax/{train,val,test}")
    print("  data/ARCADE/arcade/stenosis/{train,val,test}")


if __name__ == "__main__":
    main()
