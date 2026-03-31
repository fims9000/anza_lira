#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import shutil
from pathlib import Path

import requests


FIGSHARE_ARTICLE_ID = "19688169"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


def _pick_archive_file(files: list[dict]) -> dict:
    archives = [
        item
        for item in files
        if str(item.get("name", "")).lower().endswith((".zip", ".rar"))
    ]
    if not archives:
        raise SystemExit("The Figshare article does not expose a supported archive (.zip or .rar).")
    return max(archives, key=lambda item: int(item.get("size", 0)))


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
    seven_zip = Path(r"C:\Program Files\7-Zip\7z.exe")
    if not seven_zip.exists():
        raise SystemExit(f"7-Zip was not found at {seven_zip}; it is required to extract {zip_path.suffix} archives.")
    subprocess.run(
        [str(seven_zip), "x", str(zip_path), f"-o{out_dir}", "-y"],
        check=True,
    )


def _find_extracted_root(extract_dir: Path) -> Path:
    split_hits = list(extract_dir.rglob("train")) + list(extract_dir.rglob("Train"))
    for candidate in split_hits:
        if candidate.is_dir():
            return candidate.parent
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public FIVES archive from Figshare and prepare the raw folder for normalization.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Project data root; files will be stored under <data-root>/FIVES_raw.",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="fives_figshare_archive",
        help="Base name of the downloaded archive file; the downloader preserves the remote extension.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract even if files already exist.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_root = (project_root / args.data_root / "FIVES_raw").resolve()
    extract_dir = raw_root / "extracted"
    existing_archives = sorted(raw_root.glob(f"{args.zip_name}.*"))
    zip_path = existing_archives[0] if existing_archives else None

    if args.force and raw_root.exists():
        shutil.rmtree(raw_root)
        zip_path = None

    if zip_path is None or not zip_path.exists():
        response = requests.get(FIGSHARE_API_URL, timeout=60)
        response.raise_for_status()
        article = response.json()
        archive = _pick_archive_file(article.get("files", []))
        download_url = archive.get("download_url")
        if not download_url:
            raise SystemExit("The Figshare archive metadata does not contain a download URL.")
        archive_name = str(archive["name"])
        suffix = Path(archive_name).suffix or ".bin"
        zip_path = raw_root / f"{args.zip_name}{suffix}"
        print(f"Downloading {archive['name']} -> {zip_path}")
        _download(download_url, zip_path)
    else:
        print(f"Reusing existing archive: {zip_path}")

    if not extract_dir.exists():
        print(f"Extracting -> {extract_dir}")
        _extract(zip_path, extract_dir)
    else:
        print(f"Reusing extracted files: {extract_dir}")

    dataset_root = _find_extracted_root(extract_dir)
    print(f"FIVES raw files are ready under {dataset_root}")
    print("Next step:")
    print(rf"  python scripts\prepare_fives.py --source-root \"{dataset_root}\"")


if __name__ == "__main__":
    main()
