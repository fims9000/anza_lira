"""Strict visualMetro measurement loader: sparse expert geometry, not segmentation masks.

Source visualMetro exports contain 3-D measured positions followed by identical
point indices in the 1/3/2 film-view order. They do NOT contain complete pixel
masks, negative labels or necessarily the matching original BMP files.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any

import numpy as np

FRAME_NAME = re.compile(r"^[0-9]{3}_[0-9]{3}$")
TRACK_ID = re.compile(r"^e[0-9]+t[0-9]+(?:t[0-9]+)*$")
VIEW_ROW = re.compile(r"^\s*([123])\s+\$([0-9A-Fa-f]{8})\s+(-?\d+)\s+(-?\d+)\s*$")
FID_VIEW = re.compile(r"^\s*([123])\s+-- view\s*$")
FID_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(f\d+)\s*$")
VIEWS = (1, 3, 2)
SCHEMA = "visualmetro_sparse_measurements_v1"


class AnnotationFormatError(ValueError):
    pass


def _numbers(line: str, filename: str) -> list[float]:
    try:
        numbers = [float(x.replace("D", "E").replace("d", "E")) for x in line.split()]
    except ValueError as exc:
        raise AnnotationFormatError(f"non-numeric measurement in {filename}") from exc
    if not numbers or not bool(np.isfinite(numbers).all()):
        raise AnnotationFormatError(f"nonfinite or empty measurement in {filename}")
    return numbers


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def parse_track(path: Path, frame: str, track_id: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 6:
        raise AnnotationFormatError(f"short track file: {path.name}")
    # Two stopped tracks additionally carry a textual particle hypothesis,
    # e.g. "-1 Stop pi-". It is an operator annotation, NOT truth PID.
    header_tokens = lines[0].split()
    if len(header_tokens) < 8:
        raise AnnotationFormatError(f"short track header in {path.name}")
    header = _numbers(" ".join(header_tokens[:8]), path.name)
    if len(header) != 8 or int(header[0]) != header[0] or not 2 <= header[0] <= 4096:
        raise AnnotationFormatError(f"unexpected track header in {path.name}")
    n = int(header[0])
    expected = 1 + n + 3 * n + 1
    if len(lines) != expected:
        raise AnnotationFormatError(f"track {path.name}: expected {expected} lines, found {len(lines)}")
    geometry = np.asarray([_numbers(line, path.name) for line in lines[1:1 + n]])
    if geometry.shape != (n, 19):
        raise AnnotationFormatError(f"track {path.name}: expected {n} x 19 geometry columns")
    pixel_rows = [VIEW_ROW.fullmatch(line) for line in lines[1 + n:-1]]
    if any(match is None for match in pixel_rows):
        raise AnnotationFormatError(f"track {path.name}: malformed 2D coordinate row")
    # The historical export does not guarantee block order: source film 070
    # contains 1/3/2, 3/1/2 and 1/2/3. Do not silently relabel a view by
    # position. Each of the three n-row blocks must contain ONE distinct view.
    blocks = [pixel_rows[i * n:(i + 1) * n] for i in range(3)]
    view_order = []
    for block in blocks:
        ids = {int(row.group(1)) for row in block}
        if len(ids) != 1:
            raise AnnotationFormatError(f"track {path.name}: inconsistent view order or missing points")
        view_order.append(ids.pop())
    if sorted(view_order) != [1, 2, 3]:
        raise AnnotationFormatError(f"track {path.name}: inconsistent view order or missing points")
    views = {}
    view_color = {}
    for rows, view_id in zip(blocks, view_order):
        coords = [[int(row.group(3)), int(row.group(4))] for row in rows]
        if any(x < 0 or y < 0 for x, y in coords):
            raise AnnotationFormatError(f"track {path.name}: negative pixels in view {view_id}")
        views[str(view_id)] = coords
        view_color[str(view_id)] = sorted({row.group(2).upper() for row in rows})
    timestamp = lines[-1].strip()
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}", timestamp):
        raise AnnotationFormatError(f"track {path.name}: malformed export timestamp")
    return {
        "frame_id": frame,
        "track_id": track_id,
        "n_points": n,
        "file_sha256": _hash(path),
        "export_timestamp": timestamp,
        "header_raw_parameters": header[1:],
        "operator_note_unverified": " ".join(header_tokens[8:]) or None,
        "point_index_alignment": "corresponding indices in all 3 views are one measured 3D point",
        "xyz_cm": geometry[:, [0, 2, 4]].tolist(),
        "xyz_sigma_cm": geometry[:, [1, 3, 5]].tolist(),
        "measured_pixel_xy": views,
        "raw_pixel_row_color": view_color,
        "raw_view_block_order": view_order,
    }


def parse_fiducials(path: Path) -> dict[str, Any]:
    points: dict[str, dict[str, list[int]]] = {}
    view: int | None = None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        header = FID_VIEW.fullmatch(line)
        if header:
            view = int(header.group(1))
            if str(view) in points:
                raise AnnotationFormatError("repeated fiducial view")
            points[str(view)] = {}
            continue
        match = FID_ROW.fullmatch(line)
        if match and view is not None:
            name = match.group(3)
            if name in points[str(view)]:
                raise AnnotationFormatError(f"duplicated fiducial {name} in view {view}")
            points[str(view)][name] = [int(match.group(1)), int(match.group(2))]
    if set(points) != {"1", "2", "3"} or any(len(p) != 8 for p in points.values()):
        raise AnnotationFormatError(f"expected 8 fids x 3 views, found { {k:len(v) for k,v in points.items()} }")
    return {"file_sha256": _hash(path), "view_pixel_xy": points, "optical_coefficients": "present in source, NOT interpreted as calibrated optics"}


def read_measurement_tree(root: Path) -> dict[str, Any]:
    """Read an extracted *070* visualMetro archive without treating it as BMP labels."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise AnnotationFormatError(f"not an extracted directory: {root}")
    folders = sorted(x for x in root.iterdir() if x.is_dir() and FRAME_NAME.fullmatch(x.name))
    if not folders:
        raise AnnotationFormatError("no visualMetro frame directories")
    records = []
    all_sha = []
    for folder in folders:
        fid_path = folder / (folder.name + "_fids.txt")
        list_path = folder / (folder.name + "_trkList.txt")
        if not fid_path.is_file() or not list_path.is_file():
            raise AnnotationFormatError(f"missing fiducials or track list: {folder.name}")
        names = list_path.read_text(encoding="utf-8-sig").split()
        if len(names) != len(set(names)) or any(not TRACK_ID.fullmatch(n) for n in names):
            raise AnnotationFormatError(f"duplicate or invalid track IDs: {folder.name}")
        fiducials = parse_fiducials(fid_path)
        tracks = []
        for name in names:
            path = folder / f"{folder.name}_{name}.txt"
            if not path.is_file():
                raise AnnotationFormatError(f"missing named track: {path.name}")
            track = parse_track(path, folder.name, name)
            tracks.append(track)
            all_sha.append(track["file_sha256"])
        all_sha.extend([fiducials["file_sha256"], _hash(list_path)])
        images = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".tif", ".tiff"})
        records.append({"frame_id": folder.name, "tracks": tracks, "track_list_sha256": _hash(list_path),
                        "fiducials": fiducials, "image_filenames_present": images, "film_id": folder.name.split("_")[0]})
    film_ids = sorted({f["film_id"] for f in records})
    cnt = sum(len(f["tracks"]) for f in records)
    total = sum(t["n_points"] for f in records for t in f["tracks"])
    image_count = sum(len(f["image_filenames_present"]) for f in records)
    return {"schema": SCHEMA, "frames": records,
            "summary": {"frame_count": len(records), "film_ids": film_ids,
                        "track_count": cnt, "spatial_point_count": total,
                        "stereo_pixel_measurement_count": total * 3,
                        "fiducial_point_count": len(records) * 24,
                        "image_files_present_in_extracted_root": image_count,
                        "all_track_files_validated": True},
            "annotation_semantics": {
                "positive_annotation": "sparse expert track centerline measurement points",
                "instance_identity": "track_id within one frame, view correspondence by common point index",
                "negative_annotation": "NONE: unmeasured pixels are UNKNOWN, not background",
                "pixel_mask": "NOT PROVIDED",
                "stereo_calibration": "NOT RECONSTRUCTED",
                "source_film": "film 070 for current data; physical run/medium not certified",
            },
            "split_policy": "film/run disjoint; never split track patches or views from the same film between train/validation/test",
            "split_status": "INSUFFICIENT_INDEPENDENT_FILMS" if len(film_ids) < 3 else "REQUIRES_EXPLICIT_RUN_MAPPING",
            "manifest_digest_sha256": sha256("\n".join(sorted(all_sha)).encode()).hexdigest()}


def write_manifest(source: Path, output_path: Path) -> dict[str, Any]:
    manifest = read_measurement_tree(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
