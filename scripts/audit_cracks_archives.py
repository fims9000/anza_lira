#!/usr/bin/env python3
"""Verify, inventory, and safely extract the two official CRACKS archives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any
import zipfile
import zlib

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ARCHIVE = PROJECT_ROOT / "images.zip"
ANNOTATION_ARCHIVE = PROJECT_ROOT / "Fault segmentations.zip"
DATA_ROOT = PROJECT_ROOT / "data" / "cracks"
RESULT_ROOT = PROJECT_ROOT / "results" / "cracks_study"
EXPECTED_MD5 = {
    "images.zip": "6557236191763af7bd8298ecb136d41e",
    "Fault segmentations.zip": "01e1697e886da2079ff3c1967334a7ca",
}
SECTION_PATTERN = re.compile(r"^section_(\d+)$", re.IGNORECASE)
OFFICIAL_COLORS = {
    (255, 127, 14): "certain_no_fault",
    (44, 160, 44): "uncertain_fault",
    (31, 119, 180): "certain_fault",
}
INTERVALS = {
    "train": (1, 260),
    "guard_1": (261, 280),
    "validation": (281, 320),
    "guard_2": (321, 340),
    "test": (341, 400),
}
MINIMUM_COUNTS = {"train": 200, "validation": 25, "test": 40}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def section_id(path: str) -> int:
    match = SECTION_PATTERN.match(PurePosixPath(path).stem)
    if not match:
        raise ValueError(f"Invalid CRACKS section filename: {path}")
    return int(match.group(1))


def image_channels(mode: str) -> int:
    return int(Image.getmodebands(mode))


def _archive_record(path: Path) -> dict[str, Any]:
    actual_md5 = digest_file(path, "md5")
    expected_md5 = EXPECTED_MD5[path.name]
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        members = archive.infolist()
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "md5": actual_md5,
        "expected_md5": expected_md5,
        "md5_status": "PASS" if actual_md5 == expected_md5 else "FAIL",
        "member_count": len(members),
        "file_count": sum(not member.is_dir() for member in members),
        "corrupt_member": corrupt_member,
        "zip_crc_status": "PASS" if corrupt_member is None else "FAIL",
    }


def _assert_verified(records: dict[str, dict[str, Any]]) -> None:
    failed = [name for name, record in records.items() if record["md5_status"] != "PASS"]
    corrupt = [name for name, record in records.items() if record["zip_crc_status"] != "PASS"]
    if failed or corrupt:
        raise RuntimeError(f"STOP DATA IMPORT: md5_failures={failed}, corrupt_archives={corrupt}")


def _inspect_images() -> tuple[dict[str, Any], set[int]]:
    ids: list[int] = []
    paths: list[str] = []
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    alpha_values: Counter[int] = Counter()
    invalid_files: list[str] = []
    with zipfile.ZipFile(IMAGE_ARCHIVE) as archive:
        infos = [member for member in archive.infolist() if not member.is_dir()]
        for member in infos:
            path = PurePosixPath(member.filename)
            paths.append(member.filename)
            if path.suffix.lower() != ".png":
                invalid_files.append(member.filename)
                continue
            ids.append(section_id(member.filename))
            try:
                with Image.open(BytesIO(archive.read(member))) as image:
                    dimensions[f"{image.height}x{image.width}"] += 1
                    modes[image.mode] += 1
                    channels[str(image_channels(image.mode))] += 1
                    if "A" in image.getbands():
                        for count, value in image.getchannel("A").getcolors(256) or []:
                            alpha_values[int(value)] += int(count)
                    image.load()
            except (OSError, ValueError) as exc:
                invalid_files.append(f"{member.filename}: {exc}")
    duplicates = sorted(name for name, count in Counter(paths).items() if count > 1)
    id_duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    id_set = set(ids)
    return (
        {
            "file_count": len(paths),
            "directory_names": sorted({str(PurePosixPath(path).parent) for path in paths}),
            "extensions": dict(sorted(Counter(PurePosixPath(path).suffix.lower() for path in paths).items())),
            "dimensions_hw": dict(sorted(dimensions.items())),
            "modes": dict(sorted(modes.items())),
            "channel_counts": dict(sorted(channels.items())),
            "alpha_values": {str(key): value for key, value in sorted(alpha_values.items())},
            "image_identifiers": sorted(id_set),
            "missing_nominal_ids": [value for value in range(1, 401) if value not in id_set],
            "duplicate_paths": duplicates,
            "duplicate_section_ids": id_duplicates,
            "corrupt_or_invalid_files": invalid_files,
        },
        id_set,
    )


def _inspect_annotations() -> tuple[dict[str, Any], dict[str, Any], set[int]]:
    annotator_counts: Counter[str] = Counter()
    annotator_ids: dict[str, list[int]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    all_colors: set[tuple[int, int, int]] = set()
    expert_colors: Counter[tuple[int, int, int]] = Counter()
    paths: list[str] = []
    corrupt_or_invalid: list[str] = []
    with zipfile.ZipFile(ANNOTATION_ARCHIVE) as archive:
        infos = [member for member in archive.infolist() if not member.is_dir()]
        for member in infos:
            path = PurePosixPath(member.filename)
            paths.append(member.filename)
            extensions[path.suffix.lower()] += 1
            if len(path.parts) != 3 or path.parts[0] != "Fault segmentations" or path.suffix.lower() != ".png":
                corrupt_or_invalid.append(f"unexpected path: {member.filename}")
                continue
            annotator = path.parts[1]
            annotator_counts[annotator] += 1
            try:
                identifier = section_id(member.filename)
                annotator_ids[annotator].append(identifier)
                with Image.open(BytesIO(archive.read(member))) as image:
                    dimensions[f"{image.height}x{image.width}"] += 1
                    modes[image.mode] += 1
                    channels[str(image_channels(image.mode))] += 1
                    colors = image.convert("RGB").getcolors(maxcolors=256)
                    if colors is None:
                        raise ValueError("mask contains more than 256 colors")
                    for count, color in colors:
                        rgb = tuple(int(value) for value in color)
                        all_colors.add(rgb)
                        if annotator == "expert":
                            expert_colors[rgb] += int(count)
                    image.load()
            except (OSError, ValueError) as exc:
                corrupt_or_invalid.append(f"{member.filename}: {exc}")
    duplicate_paths = sorted(name for name, count in Counter(paths).items() if count > 1)
    duplicate_names = {
        annotator: sorted(value for value, count in Counter(ids).items() if count > 1)
        for annotator, ids in annotator_ids.items()
        if any(count > 1 for count in Counter(ids).values())
    }
    expert_ids = set(annotator_ids.get("expert", []))
    total_expert_pixels = sum(expert_colors.values())
    color_rows = []
    unknown_pixels = 0
    for rgb, count in sorted(expert_colors.items()):
        semantic = OFFICIAL_COLORS.get(rgb, "unassigned_ignore")
        if rgb not in OFFICIAL_COLORS:
            unknown_pixels += count
        color_rows.append(
            {
                "rgb": list(rgb),
                "pixel_count": count,
                "fraction": count / total_expert_pixels,
                "semantic": semantic,
                "mapping_source": "official paper/repository" if rgb in OFFICIAL_COLORS else "not documented; ignored",
            }
        )
    color_audit = {
        "status": "VERIFIED_WITH_UNASSIGNED_IGNORE",
        "official_semantics_source": {
            "repository": "https://github.com/olivesgatech/CRACKS",
            "paper": "https://arxiv.org/abs/2408.11185",
            "mapping": {
                "orange": "certain no-fault",
                "green": "uncertain fault",
                "blue": "certain fault",
            },
        },
        "colors": color_rows,
        "total_pixels": total_expert_pixels,
        "unknown_pixel_count": unknown_pixels,
        "unknown_fraction": unknown_pixels / total_expert_pixels,
        "unknown_exceeds_one_percent": unknown_pixels / total_expert_pixels > 0.01,
        "investigation": (
            "White palette index 0 has no transparency and is absent from the three official semantic colors. "
            "It is retained as unassigned/ignore and never converted to background."
        ),
        "strict_target": {
            "positive": list(next(rgb for rgb, semantic in OFFICIAL_COLORS.items() if semantic == "certain_fault")),
            "negative": list(next(rgb for rgb, semantic in OFFICIAL_COLORS.items() if semantic == "certain_no_fault")),
            "ignore": [
                list(next(rgb for rgb, semantic in OFFICIAL_COLORS.items() if semantic == "uncertain_fault")),
                *[list(rgb) for rgb in sorted(set(expert_colors) - set(OFFICIAL_COLORS))],
            ],
        },
        "inclusive_target": {
            "positive": [
                list(rgb)
                for rgb, semantic in OFFICIAL_COLORS.items()
                if semantic in {"certain_fault", "uncertain_fault"}
            ],
            "negative": list(next(rgb for rgb, semantic in OFFICIAL_COLORS.items() if semantic == "certain_no_fault")),
            "ignore": [list(rgb) for rgb in sorted(set(expert_colors) - set(OFFICIAL_COLORS))],
        },
    }
    annotation_record = {
        "file_count": len(paths),
        "annotator_directory_count": len(annotator_counts),
        "annotation_directories": dict(sorted(annotator_counts.items())),
        "expert_mask_count": len(expert_ids),
        "expert_section_ids": sorted(expert_ids),
        "extensions": dict(sorted(extensions.items())),
        "dimensions_hw": dict(sorted(dimensions.items())),
        "modes": dict(sorted(modes.items())),
        "channel_counts": dict(sorted(channels.items())),
        "unique_mask_colors_rgb": [list(color) for color in sorted(all_colors)],
        "duplicate_paths": duplicate_paths,
        "duplicate_section_ids_by_annotator": duplicate_names,
        "corrupt_or_invalid_files": corrupt_or_invalid,
        "section_ids_by_annotator": {name: sorted(set(ids)) for name, ids in sorted(annotator_ids.items())},
    }
    return annotation_record, color_audit, expert_ids


def _split_feasibility(expert_ids: set[int], image_ids: set[int]) -> dict[str, Any]:
    paired = expert_ids & image_ids
    assignments = {
        name: sorted(identifier for identifier in paired if lower <= identifier <= upper)
        for name, (lower, upper) in INTERVALS.items()
    }
    counts = {name: len(ids) for name, ids in assignments.items()}
    shortfalls = {
        name: {"actual": counts[name], "required_minimum": minimum}
        for name, minimum in MINIMUM_COUNTS.items()
        if counts[name] < minimum
    }
    return {
        "protocol": "blocked spatial intervals by numeric section ID",
        "intervals": {name: [lower, upper] for name, (lower, upper) in INTERVALS.items()},
        "minimum_counts": MINIMUM_COUNTS,
        "paired_expert_image_count": len(paired),
        "expert_without_image": sorted(expert_ids - image_ids),
        "image_without_expert": sorted(image_ids - expert_ids),
        "assignments": assignments,
        "counts": counts,
        "feasible": not shortfalls,
        "shortfalls": shortfalls,
        "training_gate": "PASS" if not shortfalls else "STOP_BEFORE_TRAINING",
        "reason": (
            None
            if not shortfalls
            else "The official archive has insufficient expert-labeled sections for the predeclared 200/25/40 blocked split."
        ),
    }


def _safe_extract(archive_path: Path, destination: Path, strip_prefix: str) -> dict[str, int]:
    extracted = 0
    verified_existing = 0
    prefix = PurePosixPath(strip_prefix)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or tuple(path.parts[: len(prefix.parts)]) != prefix.parts:
                raise ValueError(f"Unsafe or unexpected CRACKS archive member: {member.filename}")
            relative = PurePosixPath(*path.parts[len(prefix.parts) :])
            target = (destination / Path(*relative.parts)).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise ValueError(f"Archive member escapes destination: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ValueError(f"Existing extracted file has wrong size: {target}")
                crc = 0
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if crc & 0xFFFFFFFF != member.CRC:
                    raise ValueError(f"Existing extracted file has wrong CRC: {target}")
                verified_existing += 1
                continue
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    return {"extracted": extracted, "verified_existing": verified_existing}


def run_audit(*, extract: bool = False) -> dict[str, Any]:
    archive_records = {
        IMAGE_ARCHIVE.name: _archive_record(IMAGE_ARCHIVE),
        ANNOTATION_ARCHIVE.name: _archive_record(ANNOTATION_ARCHIVE),
    }
    _assert_verified(archive_records)
    image_record, image_ids = _inspect_images()
    annotation_record, color_audit, expert_ids = _inspect_annotations()
    split_feasibility = _split_feasibility(expert_ids, image_ids)
    inventory = {
        "status": "PASS",
        "archives": archive_records,
        "images": image_record,
        "annotations": annotation_record,
        "pairing": {
            "expert_with_image": sorted(expert_ids & image_ids),
            "expert_missing_image": sorted(expert_ids - image_ids),
            "images_missing_expert": sorted(image_ids - expert_ids),
        },
    }
    extraction = None
    if extract:
        extraction = {
            "images": _safe_extract(IMAGE_ARCHIVE, DATA_ROOT / "images", "images"),
            "annotations": _safe_extract(
                ANNOTATION_ARCHIVE,
                DATA_ROOT / "annotations",
                "Fault segmentations",
            ),
        }
        inventory["extraction"] = extraction
    write_json(RESULT_ROOT / "archive_inventory.json", inventory)
    write_json(RESULT_ROOT / "expert_color_audit.json", color_audit)
    write_json(RESULT_ROOT / "split_feasibility.json", split_feasibility)
    write_json(DATA_ROOT / "manifests" / "archive_inventory.json", inventory)
    write_json(DATA_ROOT / "manifests" / "expert_color_audit.json", color_audit)
    write_json(DATA_ROOT / "splits" / "split_manifest.json", split_feasibility)
    print("CRACKS ARCHIVES: VERIFIED")
    print(f"IMAGES: {image_record['file_count']}")
    print(f"ANNOTATORS: {annotation_record['annotator_directory_count']}")
    print(f"EXPERT MASKS: {annotation_record['expert_mask_count']}")
    print(f"EXPERT COLORS: {len(color_audit['colors'])}")
    print(
        "BLOCKED SPLIT: "
        f"train={split_feasibility['counts']['train']} "
        f"val={split_feasibility['counts']['validation']} "
        f"test={split_feasibility['counts']['test']}"
    )
    print(f"TRAINING GATE: {split_feasibility['training_gate']}")
    return {
        "inventory": inventory,
        "color_audit": color_audit,
        "split_feasibility": split_feasibility,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    payload = run_audit(extract=args.extract)
    return 0 if payload["inventory"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
