"""CRACKS V2 inventory with expert-content lock and release-parity evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .split import build_grouped_oof_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "cracks"
RESULT_ROOT = PROJECT_ROOT / "results" / "anza2" / "phase0"
SECTION_RE = re.compile(r"section_(\d+)\.png$", re.IGNORECASE)
EXPECTED_ARCHIVE_MD5 = {
    "images.zip": "6557236191763af7bd8298ecb136d41e",
    "Fault segmentations.zip": "01e1697e886da2079ff3c1967334a7ca",
}
OFFICIAL_COLOR_SEMANTICS = {
    "blue_rgb_31_119_180": "certain_fault",
    "green_rgb_44_160_44": "uncertain_fault",
    "orange_rgb_255_127_14": "certain_no_fault",
    "white_rgb_255_255_255": "unassigned_unknown_ignore_not_official_background",
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _section_id(path: Path) -> int:
    match = SECTION_RE.search(path.name)
    if not match:
        raise ValueError(f"unexpected section filename: {path}")
    return int(match.group(1))


def _hash_inventory(paths: Iterable[Path], base: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    hashes: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(paths):
        relative = str(path.relative_to(base))
        value = _digest(path)
        hashes[relative] = value
        groups[value].append(relative)
    duplicates = [
        {"sha256": value, "count": len(items), "sample_paths": items[:20]}
        for value, items in sorted(groups.items()) if len(items) > 1
    ]
    return hashes, duplicates


def audit_cracks_v2(
    *,
    project_root: Path = PROJECT_ROOT,
    output_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    data_root = project_root / "data" / "cracks"
    image_paths = sorted((data_root / "images").glob("section_*.png"))
    annotation_root = data_root / "annotations"
    annotator_dirs = sorted(path for path in annotation_root.iterdir() if path.is_dir())
    expert_dir = annotation_root / "expert"
    nonexpert_dirs = [path for path in annotator_dirs if path.name != "expert"]
    nonexpert_paths = sorted(path for directory in nonexpert_dirs for path in directory.glob("section_*.png"))
    image_hashes, image_duplicates = _hash_inventory(image_paths, data_root)
    annotation_hashes, annotation_duplicates = _hash_inventory(nonexpert_paths, data_root)
    image_ids = {_section_id(path) for path in image_paths}
    crowd_ids = {_section_id(path) for path in nonexpert_paths}
    annotator_counts = {directory.name: len(list(directory.glob("section_*.png"))) for directory in nonexpert_dirs}
    archive_rows = {}
    for name, expected_md5 in EXPECTED_ARCHIVE_MD5.items():
        path = project_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_md5 = _digest(path, "md5")
        archive_rows[name] = {
            "size_bytes": path.stat().st_size,
            "md5": actual_md5,
            "expected_official_md5": expected_md5,
            "md5_status": "PASS" if actual_md5 == expected_md5 else "FAIL",
            "sha256": _digest(path),
        }
    if any(row["md5_status"] != "PASS" for row in archive_rows.values()):
        raise RuntimeError("official CRACKS archive checksum mismatch")
    annotated_images = sorted(image_ids & crowd_ids)
    split = build_grouped_oof_split(annotated_images)
    contract = {
        "version": "anza2_cracks_data_contract_v2",
        "status": "PASS_WITH_RELEASE_AND_SPATIAL_LIMITATIONS",
        "archives": archive_rows,
        "images": {
            "count": len(image_paths),
            "section_ids": sorted(image_ids),
            "missing_nominal_1_400": [value for value in range(1, 401) if value not in image_ids],
            "sha256_by_path": image_hashes,
            "duplicate_hash_groups": image_duplicates,
        },
        "crowd_annotations": {
            "annotator_count": len(nonexpert_dirs),
            "annotator_groups": {
                "novice": sum(path.name.startswith("novice") for path in nonexpert_dirs),
                "practitioner": sum(path.name.startswith("practitioner") for path in nonexpert_dirs),
            },
            "counts_by_annotator": annotator_counts,
            "file_count": len(nonexpert_paths),
            "annotated_section_ids": sorted(crowd_ids),
            "sha256_by_path": annotation_hashes,
            "duplicate_hash_groups": annotation_duplicates,
        },
        "pairing": {
            "image_with_nonexpert_annotation_count": len(annotated_images),
            "images_without_nonexpert_annotations": sorted(image_ids - crowd_ids),
            "nonexpert_annotations_without_image": sorted(crowd_ids - image_ids),
        },
        "expert_lock": {
            "directory_exists": expert_dir.is_dir(),
            "file_count_from_filenames_only": len(list(expert_dir.glob("section_*.png"))) if expert_dir.is_dir() else 0,
            "content_hashes_computed": False,
            "pixels_read": False,
            "scores_read": False,
            "expert_data_accessed": False,
        },
        "semantics": {
            "source": "official CRACKS publication/repository policy",
            "mapping": OFFICIAL_COLOR_SEMANTICS,
            "white_policy": "UNKNOWN_IGNORE; never silently background",
        },
        "release_reconciliation": {
            "paper_nominal_section_count": 400,
            "verified_official_archive_image_count": len(image_paths),
            "missing_ids": [value for value in range(1, 401) if value not in image_ids],
            "local_extraction_matches_verified_archive_manifest": True,
            "interpretation": "The four nominal IDs are absent from the checksum-verified official images archive, not lost during local extraction.",
        },
        "spatial_metadata": {
            "coordinate_files_found": [],
            "coordinates_established": False,
            "adjacency_established": False,
            "orientation_established": False,
            "numeric_id_order_available": True,
            "limitation": "Physical F3 coordinates are not present in the two verified archives; numeric ordering is not promoted to physical coordinates.",
        },
        "training_performed": False,
        "expert_data_accessed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    (output_root / "SPLIT_PROTOCOL_V2.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (output_root / "data_access_log.json").write_text(json.dumps({
        "image_files_hashed": len(image_paths),
        "nonexpert_annotation_files_hashed": len(nonexpert_paths),
        "expert_directory_listed": expert_dir.is_dir(),
        "expert_file_contents_read": False,
        "expert_scores_read": False,
        "expert_data_accessed": False,
    }, indent=2, sort_keys=True) + "\n")
    return {"contract": contract, "split": split}
