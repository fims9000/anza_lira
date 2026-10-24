"""Lossless full-resolution crops of large uncompressed 8-bit grayscale BMPs.

Strict point semantics: an f1...f8 mark is a fiducial and a 'start' point is
an operator-clicked candidate vertex. Neither is a track segmentation mask.
Images are read by memory mapping rather than loading hundreds of MB in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from typing import Any
import json
import os

import numpy as np


class BitmapFormatError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CropLocation:
    original_xyxy: tuple[int, int, int, int]
    center_xy_original: tuple[float, float]
    center_xy_local: tuple[float, float]


class GrayBMP8:
    """Read only ROI pixels from a BI_RGB grayscale BMP with identity palette."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header = handle.read(54 + 256 * 4)
        if len(header) < 54 or header[:2] != b"BM":
            raise BitmapFormatError("not a Windows BMP")
        dib_size = struct.unpack_from("<I", header, 14)[0]
        if dib_size < 40:
            raise BitmapFormatError("unsupported DIB header")
        width, height_signed = struct.unpack_from("<ii", header, 18)
        planes, depth = struct.unpack_from("<HH", header, 26)
        compression = struct.unpack_from("<I", header, 30)[0]
        offset = struct.unpack_from("<I", header, 10)[0]
        if width <= 0 or height_signed == 0 or planes != 1 or depth != 8 or compression != 0:
            raise BitmapFormatError("expected uncompressed 8-bit grayscale BMP")
        if offset < 54 + 4 * 256 or len(header) < 54 + 4 * 256:
            raise BitmapFormatError("8-bit BMP palette missing")
        table = np.frombuffer(header[54:54 + 1024], dtype=np.uint8).reshape(256, 4)
        gray = np.arange(256, dtype=np.uint8)
        if not all(np.array_equal(table[:, col], gray) for col in (0, 1, 2)):
            raise BitmapFormatError("non-identity color palette; convert BMP to grayscale without resizing")
        self.width = width
        self.height = abs(height_signed)
        self.row_stride = (width + 3) // 4 * 4
        self.pixel_offset = offset
        self.top_down = height_signed < 0
        required_bytes = offset + self.row_stride * self.height
        if self.path.stat().st_size < required_bytes:
            raise BitmapFormatError("truncated BMP pixel data")
        mapped = np.memmap(self.path, dtype=np.uint8, mode="r", offset=offset,
                           shape=(self.height, self.row_stride))
        self._data = mapped if self.top_down else mapped[::-1]

    def crop(self, center_xy: tuple[float, float], size: int = 1024) -> tuple[np.ndarray, CropLocation]:
        if size < 64 or size % 16 or size > min(self.width, self.height):
            raise ValueError("tile_size must be a multiple of 16, >=64 and <= both image dimensions")
        x, y = map(float, center_xy)
        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("center not inside original BMP")
        x0 = int(round(x)) - size // 2
        y0 = int(round(y)) - size // 2
        x0 = max(0, min(x0, self.width - size))
        y0 = max(0, min(y0, self.height - size))
        x1, y1 = x0 + size, y0 + size
        image = np.asarray(self._data[y0:y1, x0:x1], dtype=np.uint8).copy()
        location = CropLocation((x0, y0, x1, y1), (x, y), (x - x0, y - y0))
        return image, location


def export_h50z_tiles(image_root: Path, labels_json: Path, output: Path, *,
                      tile_size: int = 1024, include_fiducials: bool = True) -> dict[str, Any]:
    """Export complete local annotations for available BMPs. No masks are fabricated.

    Only filenames present directly under image_root are accepted. Paths stored
    in labels.json (found_at) are ignored as untrusted metadata.
    """
    import cv2

    root = Path(image_root)
    labels = json.loads(Path(labels_json).read_text(encoding="utf-8-sig"))
    if not isinstance(labels, dict) or not labels:
        raise ValueError("keypoints JSON must be a nonempty mapping")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    frame_records = []
    for name in sorted(labels):
        if Path(name).name != name or Path(name).suffix.lower() != '.bmp':
            raise ValueError("unsafe or malformed label image name")
        path = root / name
        if not path.is_file():
            continue
        record = labels[name]
        src = GrayBMP8(path)
        if (src.width, src.height) != (int(record["original_width"]), int(record["original_height"])):
            raise ValueError(f"pixel dimensions mismatch for {name}")
        points = record["keypoints"]
        if int(record["keypoints_count"]) != len(points):
            raise ValueError(f"keypoints_count mismatch for {name}")
        allowed = {"start", *(f"f{i}" for i in range(1, 9))}
        for point in points:
            if point["label"] not in allowed:
                raise ValueError(f"unknown keypoint type for {name}: {point['label']}")
            if not (0 <= point["x_px"] < src.width and 0 <= point["y_px"] < src.height):
                raise ValueError(f"out-of-image keypoint for {name}")
        selected = [(i, point) for i, point in enumerate(points)
                    if point["label"] == "start" or (include_fiducials and point["label"] == "f4")]
        file_sha = sha256_file(path)
        frame_records.append({"image": name, "source_sha256": file_sha, "shape_yx": [src.height, src.width],
                              "all_keypoints_count": len(points),
                              "annotated_vertex_count": sum(p["label"] == "start" for p in points)})
        for ordinal, (original_index, point) in enumerate(selected):
            image, loc = src.crop((point["x_px"], point["y_px"]), tile_size)
            out_name = f"{Path(name).stem}_{point['label']}_{ordinal + 1:02d}.png"
            path_out = output / out_name
            if not cv2.imwrite(str(path_out), image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise OSError(f"could not save {path_out}")
            x0, y0, x1, y1 = loc.original_xyxy
            nearby = [{"label": p["label"], "index_in_source_json": i,
                       "xy_original_px": [p["x_px"], p["y_px"]],
                       "xy_local_px": [p["x_px"] - x0, p["y_px"] - y0]}
                      for i, p in enumerate(points)
                      if x0 <= p["x_px"] < x1 and y0 <= p["y_px"] < y1]
            entries.append({"filename": out_name, "image": name,
                            "source_image_sha256": file_sha,
                            "source_annotation_index": original_index,
                            "selected_label": point["label"],
                            "crop_xyxy_in_source": list(loc.original_xyxy),
                            "point_xy_original_px": list(loc.center_xy_original),
                            "point_xy_local_px": list(loc.center_xy_local),
                            "all_keypoints_visible_in_tile": nearby,
                            "tile_sha256": sha256_file(path_out)})
    if not frame_records:
        raise ValueError("no BMP file matching labels JSON found")
    manifest = {"schema": "bubble_h50z_partial_keypoints_v1", "available_source_images": len(frame_records),
                "images_mentioned_in_json": len(labels), "frames": frame_records, "tiles": entries,
                "supervision": "FIDUCIALS_AND_START_KEYPOINTS_ONLY_NOT_FULL_TRACK_MASKS",
                "unmarked_pixels": "UNLABELED_NOT_TRUE_NEGATIVE",
                "spatial_sampling": "lossless original-pixel resolution; x/y offsets preserve provenance",
                "data_split": "split by physical run/film, never by tiles from the same source"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    return manifest


def export_visualmetro_triplet_tiles(
    track: dict[str, Any], source_bmps: dict[int, Path], output: Path,
    *, indices: tuple[int, ...] = (0,), tile_size: int = 1024,
) -> dict[str, Any]:
    """Export matched 3-view patches from existing visualMetro track measurements.

    The track's common point index is the stereo correspondence, but *matching
    raw BMP identity* is the operator's responsibility until verified by optics.
    """
    import cv2

    if set(source_bmps) != {1, 2, 3}:
        raise ValueError("three explicit view paths (1,2,3) required")
    if not track.get('n_points') or any(not 0 <= i < track['n_points'] for i in indices):
        raise ValueError("point index is not present in the track")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate point index")
    bmp = {view: GrayBMP8(path) for view, path in source_bmps.items()}
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    sources = {str(v): {"filename": source_bmps[v].name, "sha256": sha256_file(source_bmps[v]),
                        "width": bmp[v].width, "height": bmp[v].height}
               for v in (1, 2, 3)}
    # Points AND all fiducials should be checked before creating crops, in the
    # full visualMetro pipeline. Here only the track positions are available.
    for view in (1,2,3):
        for x, y in track['measured_pixel_xy'][str(view)]:
            if not (0 <= x < bmp[view].width and 0 <= y < bmp[view].height):
                raise ValueError(f"track contains pixel outside view {view} BMP; wrong image likely")
    entries = []
    for idx in indices:
        entry = {"frame_id": track['frame_id'], "track_id": track['track_id'],
                 "common_measured_point_index": idx, "xyz_cm_from_visualmetro": track['xyz_cm'][idx],
                 "views": {}}
        for view in (1, 2, 3):
            xy = track['measured_pixel_xy'][str(view)][idx]
            tile, loc = bmp[view].crop(tuple(xy), size=tile_size)
            out_name = f"{track['frame_id']}_{track['track_id']}_v{view}_p{idx:03d}.png"
            out_path = output / out_name
            if not cv2.imwrite(str(out_path), tile, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise OSError(f"could not save {out_path}")
            x0, y0, x1, y1 = loc.original_xyxy
            visible = [{"index": j, "xy_original_px": p,
                        "xy_local_px": [p[0]-x0,p[1]-y0]}
                       for j,p in enumerate(track['measured_pixel_xy'][str(view)])
                       if x0<=p[0]<x1 and y0<=p[1]<y1]
            entry['views'][str(view)] = {"filename": out_name,
                                          "point_xy_original_px": xy,
                                          "point_xy_local_px": list(loc.center_xy_local),
                                          "crop_xyxy_in_source": list(loc.original_xyxy),
                                          "other_measured_points_in_tile": visible,
                                          "sha256": sha256_file(out_path)}
        entries.append(entry)
    result = {"schema": "visualmetro_matched_sparse_triplet_patches_v1",
              "frame_id": track['frame_id'], "track_id": track['track_id'],
              "source_bmp": sources, "selected_correspondences": entries,
              "source_track_sha256": track['file_sha256'],
              "annotation_semantics": "SPARSE_MEASURED_TRACK_COORDINATES_NO_DENSE_FOREGROUND_MASK",
              "original_resolution": True, "crop_size_px": tile_size,
              "camera_geometry_calibration_verified": False,
              "no_background_labels": True,
              "film_group_identity": track['frame_id'].split('_')[0],
              "split_policy": "do not place different views/patches of the same film into different splits"}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
