"""Strict input contract for *three-view* bubble-chamber film acquisitions.

No stereo geometry or physical particle properties are inferred here. The
manifest pins raw photographs, calibration file references, and run-disjoint
splits. Three unrelated pictures must never be treated as an event.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = "bubble_stereo_dataset_v1"
REQUIRED = ("event_id", "chamber", "run_id", "film_id", "frame_id", "view1", "view2", "view3")
CALIBRATIONS = ("optical_file", "fiducial_file", "magnetic_field_file")


class InputContractError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def within_root(root: Path, name: str) -> Path:
    if not name or not name.strip():
        raise InputContractError("empty file path")
    path = Path(name)
    if path.is_absolute():
        raise InputContractError(f"expected root-relative file path, got {name}")
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise InputContractError("file path escapes data root: " + name)
    if not candidate.is_file():
        raise InputContractError("file does not exist: " + name)
    return candidate


def read_film_bmp(path: Path) -> np.ndarray:
    if path.suffix.lower() != ".bmp":
        raise InputContractError(f"expected original BMP file: {path.name}")
    with path.open("rb") as fd:
        if fd.read(2) != b"BM":
            raise InputContractError("not a valid Windows BMP signature: " + path.name)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 64:
        raise InputContractError("invalid or tiny BMP photo: " + path.name)
    return image


def _photo_record(root: Path, name: str, view_id: int) -> dict[str, Any]:
    path = within_root(root, name)
    image = read_film_bmp(path)
    return {"view_id": int(view_id), "file": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path), "shape_yx": list(image.shape),
            "bytes": int(path.stat().st_size)}


def build_manifest(csv_path: Path, image_root: Path, *, seed: int = 20261024) -> dict:
    root = image_root.resolve()
    csv_path = csv_path.resolve()
    if not root.is_dir() or not csv_path.is_file():
        raise InputContractError("invalid image root or acquisition CSV")
    with csv_path.open(encoding="utf-8-sig", newline="") as fd:
        reader = csv.DictReader(fd)
        if not reader.fieldnames or not set(REQUIRED).issubset(reader.fieldnames):
            raise InputContractError("acquisition CSV must include " + ", ".join(REQUIRED))
        rows = list(reader)
    if not rows:
        raise InputContractError("empty acquisition table")
    events: list[dict] = []
    used_paths: set[str] = set()
    event_ids: set[str] = set()
    for row in rows:
        fields = {key: (row.get(key) or "").strip() for key in REQUIRED}
        if any(not fields[k] for k in REQUIRED):
            raise InputContractError("missing required acquisition field")
        event_id = fields["event_id"]
        if event_id in event_ids:
            raise InputContractError("duplicate event_id: " + event_id)
        event_ids.add(event_id)
        views = [_photo_record(root, fields[f"view{i}"], i) for i in range(1, 4)]
        if len({view["sha256"] for view in views}) < 3:
            raise InputContractError("identical BMP payloads used as multiple views")
        for view in views:
            if view["file"] in used_paths:
                raise InputContractError("BMP reused by multiple views or events: " + view["file"])
            used_paths.add(view["file"])
        calibrations = {}
        for key in CALIBRATIONS:
            ref = (row.get(key) or "").strip()
            calibrations[key] = {"file": ref, "sha256": file_sha256(within_root(root, ref))} if ref else None
        events.append({"event_id": event_id, "chamber": fields["chamber"],
                       "run_id": fields["run_id"], "film_id": fields["film_id"],
                       "frame_id": fields["frame_id"], "group_id": f'{fields["chamber"]}:{fields["run_id"]}',
                       "views": views, "calibration_references": calibrations,
                       "calibration_status": "FILES_PRESENT_UNVERIFIED" if all(calibrations.values()) else "INCOMPLETE",
                       "stereo_geometry_status": "NOT_RECONSTRUCTED"})
    groups = sorted({e["group_id"] for e in events}, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    if len(groups) >= 3:
        n_train = max(1, min(len(groups) - 2, int(0.6 * len(groups))))
        n_val = max(1, min(len(groups) - n_train - 1, int(0.2 * len(groups))))
        split_by_group = {name: ("train" if i < n_train else "validation" if i < n_train + n_val else "test") for i, name in enumerate(groups)}
        split_status = "RUN_DISJOINT_ASSIGNED"
    else:
        split_by_group = {name: "unassigned" for name in groups}
        split_status = "INSUFFICIENT_RUN_GROUPS_FOR_HOLDOUT"
    for event in events:
        event["split"] = split_by_group[event["group_id"]]
    return {"schema": SCHEMA, "seed": seed, "root_path": str(root),
            "source_csv_sha256": file_sha256(csv_path), "event_count": len(events),
            "run_group_count": len(groups), "split_status": split_status,
            "grouping_rule": "entire chamber+run_id stays in one split; do not split individual photographs",
            "events": events, "no_particle_identity_labels": True,
            "image_measurement_status": "UNLABELED_ONLY",
            "physical_reconstruction_status": "NOT_RUN"}


def get_event(manifest: dict, event_id: str) -> dict:
    if manifest.get("schema") != SCHEMA:
        raise InputContractError("unsupported manifest schema")
    found = [e for e in manifest.get("events", []) if e["event_id"] == event_id]
    if len(found) != 1:
        raise InputContractError("event not found or ambiguous: " + event_id)
    return found[0]


def verified_triplet(manifest: dict, event_id: str) -> list[np.ndarray]:
    event = get_event(manifest, event_id)
    root = Path(manifest["root_path"]).resolve()
    if len(event.get("views", [])) != 3 or [v["view_id"] for v in event["views"]] != [1, 2, 3]:
        raise InputContractError("exactly views 1,2,3 required")
    for key, ref in event.get("calibration_references", {}).items():
        if ref is not None:
            path = within_root(root, ref["file"])
            if file_sha256(path) != ref["sha256"]:
                raise InputContractError("calibration reference checksum changed: " + key)
    image_views = []
    for view in event["views"]:
        path = within_root(root, view["file"])
        if file_sha256(path) != view["sha256"]:
            raise InputContractError("photo checksum changed: " + view["file"])
        image = read_film_bmp(path)
        if list(image.shape) != view["shape_yx"]:
            raise InputContractError("photo dimensions changed")
        image_views.append(image)
    return image_views
