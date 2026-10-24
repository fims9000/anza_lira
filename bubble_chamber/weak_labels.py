"""Convert expert visualMetro points into explicitly *partial* image annotations.

The clicked points are trusted foreground landmarks; interpolated curves are
only graphical guides and MUST NOT be treated as exhaustive binary masks.
The rest of the film is UNKNOWN: other unmeasured charged tracks may exist.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import csv
import json
from typing import Any

import cv2
import numpy as np

from .visualmetro_import import AnnotationFormatError, SCHEMA


class AnnotationAlignmentError(AnnotationFormatError):
    pass


def validate_view_alignment(frame: dict, view_id: int, image_shape_yx: tuple[int, int]) -> dict:
    """Reject swapped/incompatible view BMPs without silently resizing/rescaling.

    This validates pixel bounds only, NOT optical geometry or true stereo
    correspondence. The fiducial positions are geometric anchors to inspect.
    """
    v = str(view_id)
    if v not in {"1", "2", "3"}:
        raise AnnotationAlignmentError("view must be 1, 2 or 3")
    h, w = image_shape_yx
    if h < 64 or w < 64:
        raise AnnotationAlignmentError("image too small")
    fiducials = frame["fiducials"]["view_pixel_xy"][v]
    clicked = sum((track["measured_pixel_xy"][v] for track in frame["tracks"]), [])
    all_xy = np.asarray(clicked + list(fiducials.values()), dtype=np.int64)
    if all_xy.ndim != 2 or all_xy.shape[1] != 2:
        raise AnnotationAlignmentError("missing clicked points and fiducials")
    valid = (all_xy[:, 0] >= 0) & (all_xy[:, 0] < w) & (all_xy[:, 1] >= 0) & (all_xy[:, 1] < h)
    if not np.all(valid):
        first = all_xy[np.flatnonzero(~valid)[0]].tolist()
        raise AnnotationAlignmentError(f"frame {frame['frame_id']} view {v}: annotation at {first} is outside image {w}x{h}")
    return {"frame_id": frame["frame_id"], "view_id": view_id,
            "image_shape_yx": [h, w], "measured_points_inside_image": len(clicked),
            "fiducials_inside_image": len(fiducials),
            "status": "PIXEL_BOUNDS_VALID_STEREO_CALIBRATION_UNVERIFIED"}


def sparse_patch(
    image: np.ndarray,
    frame: dict,
    view_id: int,
    center_xy: tuple[int, int],
    *,
    patch_size: int = 256,
    point_radius_px: int = 2,
) -> dict[str, Any]:
    """Create a crop with observed clicked pixels, no fabricated negatives.

    Dense line_guide is a visualization only. Points outside the crop are not
    included. A `known_negative` mask is intentionally NOT produced.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("grayscale uint8 image required")
    if patch_size < 32 or patch_size % 16:
        raise ValueError("patch size must be >=32 and divisible by 16")
    if point_radius_px < 0 or point_radius_px > 6:
        raise ValueError("invalid clicked-point radius")
    validate_view_alignment(frame, view_id, image.shape)
    h, w = image.shape
    cx, cy = map(int, center_xy)
    if not (0 <= cx < w and 0 <= cy < h):
        raise AnnotationAlignmentError("patch center outside image")
    x0, y0 = cx - patch_size // 2, cy - patch_size // 2
    x1, y1 = x0 + patch_size, y0 + patch_size
    # Padding is unknown image content, not trusted negative supervision.
    actual_x0, actual_x1 = max(x0, 0), min(x1, w)
    actual_y0, actual_y1 = max(y0, 0), min(y1, h)
    pixels = np.full((patch_size, patch_size), int(np.median(image)), dtype=np.uint8)
    observed = np.zeros_like(pixels, dtype=np.uint8)
    region = image[actual_y0:actual_y1, actual_x0:actual_x1]
    yy0, yy1 = actual_y0 - y0, actual_y1 - y0
    xx0, xx1 = actual_x0 - x0, actual_x1 - x0
    pixels[yy0:yy1, xx0:xx1] = region
    observed[yy0:yy1, xx0:xx1] = 1
    positive = np.zeros_like(observed, dtype=np.uint8)
    guide = np.zeros_like(observed, dtype=np.uint8)
    ids = []
    for track in frame["tracks"]:
        xy = np.asarray(track["measured_pixel_xy"][str(view_id)], dtype=np.int32)
        local_xy = xy - [x0, y0]
        inside = (local_xy[:, 0] >= 0) & (local_xy[:, 0] < patch_size) & (local_xy[:, 1] >= 0) & (local_xy[:, 1] < patch_size)
        if np.any(inside):
            ids.append(track["track_id"])
        for x, y in local_xy[inside]:
            cv2.circle(positive, (int(x), int(y)), point_radius_px, 1, thickness=-1)
        # DO NOT bridge arbitrary long missing regions into supervision.
        # This path is only a visual approximation to sparse human measurements.
        if len(local_xy) >= 2:
            cv2.polylines(guide, [local_xy.reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8)
    positive &= observed
    guide &= observed
    return {"image": pixels, "clicked_foreground": positive, "visual_line_guide_untrusted": guide,
            "observed_pixels": observed,
            "roi_xyxy_original": [x0, y0, x1, y1],
            "containing_track_ids": ids,
            "label_semantics": "POSITIVE_SCRIBBLES_ONLY_ALL_OTHER_PIXELS_UNLABELED",
            "negative_supervision": "UNAVAILABLE",
            "dense_segmentation_ground_truth": False}


def pairing_table(manifest: dict) -> list[dict[str, Any]]:
    """Build expert-approved stereo correspondence tuples (NOT physical fits)."""
    if manifest.get("schema") != SCHEMA:
        raise AnnotationFormatError("unsupported annotation manifest")
    tuples = []
    for frame in manifest["frames"]:
        for track in frame["tracks"]:
            for idx in range(track["n_points"]):
                row: dict[str, Any] = {"film_id": frame["film_id"], "frame_id": frame["frame_id"],
                                      "track_id": track["track_id"], "point_index": idx,
                                      "x_cm": track["xyz_cm"][idx][0],
                                      "y_cm": track["xyz_cm"][idx][1],
                                      "z_cm": track["xyz_cm"][idx][2]}
                for v in (1, 3, 2):
                    x, y = track["measured_pixel_xy"][str(v)][idx]
                    row[f"v{v}_x_px"] = x
                    row[f"v{v}_y_px"] = y
                tuples.append(row)
    return tuples


def export_pairs_csv(manifest: dict, out_path: Path) -> None:
    rows = pairing_table(manifest)
    if not rows:
        raise AnnotationFormatError("no stereo pairs")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attach_bmp_inventory(manifest: dict, map_csv: Path, image_root: Path) -> list[dict]:
    """Attach exact image files by explicit frame+view, never by guessed names.

    Mapping CSV columns: frame_id,view_id,file. The function scans each file,
    validates that all fiducials and measurement points fit its pixel array,
    and pins it by SHA256. Does not assume negative masks from unmarked areas.
    """
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise AnnotationAlignmentError("image root missing")
    with map_csv.open(newline="", encoding="utf-8-sig") as fd:
        reader = csv.DictReader(fd)
        if reader.fieldnames is None or not {"frame_id", "view_id", "file"}.issubset(reader.fieldnames):
            raise AnnotationAlignmentError("mapping must have frame_id,view_id,file")
        rows = list(reader)
    by_frame = {frame["frame_id"]: frame for frame in manifest["frames"]}
    seen = set(); out = []
    for row in rows:
        frame_id, view, relative = row["frame_id"].strip(), int(row["view_id"]), row["file"].strip()
        if (frame_id, view) in seen or frame_id not in by_frame or view not in (1, 2, 3):
            raise AnnotationAlignmentError("unknown frame/view or duplicate mapping row")
        seen.add((frame_id, view))
        if not relative or Path(relative).is_absolute():
            raise AnnotationAlignmentError("expected root-relative BMP filename")
        path = (image_root / relative).resolve()
        if not path.is_relative_to(image_root) or path.suffix.lower() != '.bmp' or not path.is_file():
            raise AnnotationAlignmentError("invalid BMP reference: " + relative)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.dtype != np.uint8:
            raise AnnotationAlignmentError("BMP decoding failed: " + relative)
        check = validate_view_alignment(by_frame[frame_id], view, image.shape)
        check.update({"file": relative, "sha256": sha256(path.read_bytes()).hexdigest()})
        out.append(check)
    expected = {(frame["frame_id"], v) for frame in manifest["frames"] for v in (1, 2, 3)}
    if seen != expected:
        raise AnnotationAlignmentError(f"missing BMP mappings: {sorted(expected - seen)}")
    return out
