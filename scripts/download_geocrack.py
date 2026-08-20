#!/usr/bin/env python3
"""Download and verify only official GeoCrack patched data from Dataverse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
from typing import Any, Iterable
import zipfile

import numpy as np
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.geocrack import discover_pairs, sha256_file


DEFAULT_SERVER = "https://dataverse.harvard.edu"
DEFAULT_PID = "doi:10.7910/DVN/E4OXHQ"
EXPECTED_PAIR_COUNT = 12158
INCOMPLETE_SUFFIXES = (".crdownload", ".part", ".tmp")


def fetch_metadata(
    server: str,
    persistent_id: str,
    *,
    attempts: int = 3,
    timeout: float = 120.0,
    initial_delay: float = 5.0,
) -> dict[str, Any]:
    url = f"{server.rstrip('/')}/api/datasets/:persistentId/"
    delay = initial_delay
    last_status: int | None = None
    waf_action: str | None = None
    for attempt in range(1, attempts + 1):
        response = requests.get(url, params={"persistentId": persistent_id}, timeout=timeout)
        last_status = response.status_code
        waf_action = response.headers.get("x-amzn-waf-action")
        if response.status_code == 200:
            payload = response.json()
            if payload.get("status") != "OK" or "data" not in payload:
                raise ValueError("Dataverse returned malformed GeoCrack metadata")
            return payload
        if response.status_code != 202:
            response.raise_for_status()
        if attempt < attempts:
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError(
        f"Dataverse metadata remained unavailable after {attempts} attempts "
        f"(last HTTP status {last_status}, WAF action {waf_action or 'none'}) for {persistent_id}"
    )


def select_patched_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        files = metadata["data"]["latestVersion"]["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Dataverse metadata has no latestVersion.files") from exc
    selected = []
    for item in files:
        data_file = item.get("dataFile", {})
        filename = str(data_file.get("filename", ""))
        directory = str(item.get("directoryLabel", ""))
        searchable = f"{directory}/{filename}".lower()
        if "patched data" in searchable or "patched_data" in searchable or "patched-data" in searchable:
            if not data_file.get("id"):
                raise ValueError(f"Patched Dataverse file lacks numeric id: {filename}")
            selected.append(item)
    if not selected:
        raise ValueError("No files labeled as GeoCrack Patched Data in Dataverse metadata")
    return selected


def _verify_dataverse_checksum(path: Path, checksum: dict[str, str] | None) -> None:
    if not checksum:
        return
    algorithm = str(checksum.get("type", "")).replace("-", "").lower()
    expected = str(checksum.get("value", "")).lower()
    if not algorithm or not expected:
        return
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported Dataverse checksum algorithm: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {digest.hexdigest()}")


def download_file(
    server: str,
    item: dict[str, Any],
    destination: Path,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data_file = item["dataFile"]
    expected_size = int(data_file.get("filesize", 0))
    checksum = data_file.get("checksum")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_size or destination.stat().st_size == expected_size):
        _verify_dataverse_checksum(destination, checksum)
        return {"status": "already_verified", "sha256": sha256_file(destination)}

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    url = f"{server.rstrip('/')}/api/access/datafile/{data_file['id']}"
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        append = bool(existing and response.status_code == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if expected_size and partial.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {destination.name}: expected {expected_size}, got {partial.stat().st_size}")
    _verify_dataverse_checksum(partial, checksum)
    partial.replace(destination)
    return {"status": "downloaded", "sha256": sha256_file(destination)}


def safe_extract_zip(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in {archive}: {member.filename}")
            target = destination.joinpath(*member_path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            with handle.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    return extracted


def _reject_incomplete_downloads(path: Path) -> None:
    candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    incomplete = [item for item in candidates if item.name.lower().endswith(INCOMPLETE_SUFFIXES)]
    if incomplete:
        names = ", ".join(str(item) for item in incomplete[:5])
        raise RuntimeError(f"Incomplete browser download marker found; import refused: {names}")


def _path_snapshot(path: Path) -> tuple[tuple[str, int, int], ...]:
    if path.is_file():
        stat = path.stat()
        return ((path.name, stat.st_size, stat.st_mtime_ns),)
    return tuple(
        (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def assert_download_stable(path: Path, *, stability_seconds: float = 2.0) -> None:
    """Fail closed when a local archive/tree is absent, partial, empty, or changing."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    _reject_incomplete_downloads(path)
    before = _path_snapshot(path)
    if not before or any(size <= 0 for _, size, _ in before):
        raise RuntimeError(f"Local GeoCrack input is empty or contains empty files: {path}")
    if stability_seconds > 0:
        time.sleep(stability_seconds)
    after = _path_snapshot(path)
    if before != after:
        raise RuntimeError(f"Local GeoCrack input is still changing; import refused: {path}")


def _safe_extract_tar(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    root = destination.resolve()
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed in {archive}: {member.name}")
        for member in members:
            if not member.isfile():
                continue
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.size:
                continue
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    return extracted


def _validate_complete_archive(archive: Path) -> str:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            corrupt = handle.testzip()
            if corrupt is not None:
                raise ValueError(f"Corrupt ZIP member in {archive}: {corrupt}")
        return "zip"
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            if not any(member.isfile() for member in handle.getmembers()):
                raise ValueError(f"TAR archive has no files: {archive}")
        return "tar"
    raise ValueError(f"Unsupported or incomplete GeoCrack archive format: {archive}")


def _pair_checksums(root: Path, pairs: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    aggregate = hashlib.sha256()
    records: list[dict[str, str]] = []
    for pair in pairs:
        record = {"patch_id": pair["patch_id"], "source_image_id": pair["source_image_id"]}
        for key in ("image_path", "mask_path"):
            relative = pair[key]
            digest = sha256_file(root / relative)
            record[key] = relative
            record[f"{key}_sha256"] = digest
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
        records.append(record)
    return records, aggregate.hexdigest()


def validate_local_pairs(root: Path, *, expected_pairs: int | None = EXPECTED_PAIR_COUNT) -> dict[str, Any]:
    """Validate pairing, dimensions/readability, strict mask values, and checksums."""
    root = root.resolve()
    pairs = discover_pairs(root)
    if expected_pairs is not None and len(pairs) != int(expected_pairs):
        raise ValueError(f"Expected {expected_pairs} GeoCrack pairs, found {len(pairs)} under {root}")
    patch_ids = [pair["patch_id"] for pair in pairs]
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError("GeoCrack patch IDs are not unique")
    for pair in pairs:
        image_path = root / pair["image_path"]
        mask_path = root / pair["mask_path"]
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                mask_size = mask.size
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable GeoCrack pair {pair['patch_id']}: {exc}") from exc
        if image_size != mask_size:
            raise ValueError(f"Image/mask size mismatch for {pair['patch_id']}: {image_size} != {mask_size}")
        values = set(int(value) for value in np.unique(mask_array))
        if not values.issubset({0, 1, 255}):
            raise ValueError(f"Non-binary mask for {pair['patch_id']}: values={sorted(values)[:10]}")
    records, dataset_sha256 = _pair_checksums(root, pairs)
    return {
        "status": "PASS",
        "root": str(root),
        "pair_count": len(pairs),
        "unique_patch_count": len(set(patch_ids)),
        "source_image_count": len({pair["source_image_id"] for pair in pairs}),
        "dataset_sha256": dataset_sha256,
        "pairs": records,
    }


def import_local_geocrack(
    output_root: Path,
    *,
    local_archive: Path | None = None,
    local_root: Path | None = None,
    expected_pairs: int | None = EXPECTED_PAIR_COUNT,
    stability_seconds: float = 2.0,
) -> dict[str, Any]:
    """Import a completed browser download without contacting Dataverse."""
    if (local_archive is None) == (local_root is None):
        raise ValueError("Provide exactly one of local_archive or local_root")
    source = (local_archive or local_root)
    assert source is not None
    assert_download_stable(source, stability_seconds=stability_seconds)
    output_root.mkdir(parents=True, exist_ok=True)
    input_kind = "directory"
    input_sha256: str | None = None
    extracted_file_count = 0
    data_root = source.resolve()
    if local_archive is not None:
        input_kind = _validate_complete_archive(source)
        input_sha256 = sha256_file(source)
        data_root = (output_root / "patched_data").resolve()
        previous_path = output_root / "manual_import_manifest.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous.get("input_sha256") == input_sha256 and Path(previous.get("data_root", "")) == data_root:
                validation = validate_local_pairs(data_root, expected_pairs=expected_pairs)
                return {**previous, "status": "already_verified", "validation": validation}
        if data_root.exists() and any(data_root.iterdir()):
            raise RuntimeError(
                f"Refusing to mix a new archive with existing extracted files in {data_root}; "
                "move the old tree explicitly before importing"
            )
        if input_kind == "zip":
            extracted_file_count = safe_extract_zip(source, data_root)
        else:
            extracted_file_count = _safe_extract_tar(source, data_root)
    validation = validate_local_pairs(data_root, expected_pairs=expected_pairs)
    manifest = {
        "status": "PASS",
        "mode": "manual_local_import",
        "input_kind": input_kind,
        "input_path": str(source.resolve()),
        "input_sha256": input_sha256,
        "data_root": str(data_root),
        "extracted_file_count": extracted_file_count,
        "validation": validation,
        "network_requests": 0,
    }
    (output_root / "manual_import_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def download_geocrack(
    output_root: Path,
    *,
    server: str = DEFAULT_SERVER,
    persistent_id: str = DEFAULT_PID,
    metadata_file: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if metadata_file:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        metadata = fetch_metadata(server, persistent_id)
    metadata_path = output_root / "dataverse_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = []
    archives_dir = output_root / "archives"
    patched_dir = output_root / "patched_data"
    for item in select_patched_files(metadata):
        data_file = item["dataFile"]
        filename = Path(str(data_file["filename"])).name
        destination = archives_dir / filename
        result = download_file(server, item, destination)
        record = {
            "datafile_id": data_file["id"],
            "filename": filename,
            "directory_label": item.get("directoryLabel"),
            "size": destination.stat().st_size,
            **result,
        }
        if zipfile.is_zipfile(destination):
            record["extracted_file_count"] = safe_extract_zip(destination, patched_dir)
        records.append(record)
    manifest = {
        "persistent_id": persistent_id,
        "server": server,
        "metadata_sha256": sha256_file(metadata_path),
        "files": records,
    }
    (output_root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/geocrack"))
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--persistent-id", default=DEFAULT_PID)
    parser.add_argument("--metadata-file", type=Path, help="Use saved official metadata instead of querying the API")
    local = parser.add_mutually_exclusive_group()
    local.add_argument("--local-archive", type=Path, help="Import a completed local ZIP/TAR browser download")
    local.add_argument("--local-root", type=Path, help="Validate an already extracted local GeoCrack directory")
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_PAIR_COUNT)
    parser.add_argument("--stability-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.local_archive or args.local_root:
        manifest = import_local_geocrack(
            args.output_root,
            local_archive=args.local_archive,
            local_root=args.local_root,
            expected_pairs=args.expected_pairs,
            stability_seconds=args.stability_seconds,
        )
        print(f"PAIRS: {manifest['validation']['pair_count']}")
        print(f"DATASET SHA256: {manifest['validation']['dataset_sha256']}")
    else:
        manifest = download_geocrack(
            args.output_root,
            server=args.server,
            persistent_id=args.persistent_id,
            metadata_file=args.metadata_file,
        )
        print(f"PATCHED FILES: {len(manifest['files'])}")
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
