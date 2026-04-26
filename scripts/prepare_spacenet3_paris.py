from __future__ import annotations

import argparse
import io
import json
import random
import re
import tarfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile
from PIL import Image


IMAGE_ID_RE = re.compile(r"_img(\d+)\.tif$", re.IGNORECASE)
GEO_ID_RE = re.compile(r"_img(\d+)\.geojson$", re.IGNORECASE)


def _member_id(member_name: str, pattern: re.Pattern[str]) -> int | None:
    m = pattern.search(member_name)
    if not m:
        return None
    return int(m.group(1))


def _collect_members(
    tf: tarfile.TarFile,
    suffix: str,
    pattern: re.Pattern[str],
) -> dict[int, tarfile.TarInfo]:
    out: dict[int, tarfile.TarInfo] = {}
    for member in tf.getmembers():
        if not member.isfile():
            continue
        name = member.name
        if not name.lower().endswith(suffix):
            continue
        sample_id = _member_id(name, pattern)
        if sample_id is None:
            continue
        out[sample_id] = member
    return out


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in {3, 4, 8} and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Unexpected image shape for TIFF: {arr.shape}")

    if arr.shape[2] >= 5:
        # WorldView-like PS-MS: use Red, Green, Blue channels.
        rgb = np.stack([arr[:, :, 4], arr[:, :, 2], arr[:, :, 1]], axis=2).astype(np.float32)
    elif arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.float32)
    else:
        rgb = np.repeat(arr[:, :, :1].astype(np.float32), 3, axis=2)

    out = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        band = rgb[:, :, c]
        lo = float(np.percentile(band, 2.0))
        hi = float(np.percentile(band, 98.0))
        if hi <= lo:
            out[:, :, c] = np.clip(band, 0, 255).astype(np.uint8)
            continue
        scaled = (band - lo) / (hi - lo)
        out[:, :, c] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    return out


def _extract_geotags(page: tifffile.TiffPage) -> tuple[float, float, float, float]:
    tie = page.tags["ModelTiepointTag"].value
    scale = page.tags["ModelPixelScaleTag"].value
    i0 = float(tie[0])
    j0 = float(tie[1])
    lon0 = float(tie[3])
    lat0 = float(tie[4])
    sx = float(scale[0])
    sy = float(scale[1])
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError(f"Invalid pixel scale: sx={sx}, sy={sy}")
    return i0, j0, lon0, lat0, sx, sy


def _lonlat_to_pixel(
    lon: float,
    lat: float,
    i0: float,
    j0: float,
    lon0: float,
    lat0: float,
    sx: float,
    sy: float,
) -> tuple[int, int]:
    # North-up geotiff: x increases to the east, y increases downward.
    x = (lon - lon0) / sx + i0
    y = (lat0 - lat) / sy + j0
    return int(round(x)), int(round(y))


def _iter_lines(features: Iterable[dict]) -> Iterable[list[tuple[float, float]]]:
    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = str(geom.get("type", ""))
        coords = geom.get("coordinates")
        if not coords:
            continue
        if gtype == "LineString":
            yield [(float(p[0]), float(p[1])) for p in coords if len(p) >= 2]
        elif gtype == "MultiLineString":
            for line in coords:
                yield [(float(p[0]), float(p[1])) for p in line if len(p) >= 2]


def _rasterize_mask(
    geojson_payload: dict,
    height: int,
    width: int,
    geotags: tuple[float, float, float, float, float, float],
    line_width: int,
) -> np.ndarray:
    i0, j0, lon0, lat0, sx, sy = geotags
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in _iter_lines(geojson_payload.get("features", [])):
        pts: list[tuple[int, int]] = []
        for lon, lat in line:
            x, y = _lonlat_to_pixel(lon, lat, i0, j0, lon0, lat0, sx, sy)
            if x < -width or x > 2 * width or y < -height or y > 2 * height:
                continue
            pts.append((x, y))
        if len(pts) >= 2:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [poly], isClosed=False, color=255, thickness=int(line_width), lineType=cv2.LINE_AA)
    return (mask > 127).astype(np.uint8) * 255


def _split_ids(ids: list[int], seed: int, val_fraction: float, test_fraction: float) -> dict[int, str]:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_test + n_val >= n:
        n_test = max(1, min(n_test, n - 2))
        n_val = max(1, min(n_val, n - n_test - 1))
    test_ids = set(shuffled[:n_test])
    val_ids = set(shuffled[n_test : n_test + n_val])
    mapping: dict[int, str] = {}
    for sid in shuffled:
        if sid in test_ids:
            mapping[sid] = "in-domain-test"
        elif sid in val_ids:
            mapping[sid] = "val"
        else:
            mapping[sid] = "train"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpaceNet3 AOI_3_Paris into GlobalScaleRoad-like split.")
    parser.add_argument(
        "--image-tar",
        default="data/SpaceNet3_Roads/tarballs/SN3_roads_train_AOI_3_Paris.tar.gz",
    )
    parser.add_argument(
        "--geojson-tar",
        default="data/SpaceNet3_Roads/tarballs/SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz",
    )
    parser.add_argument(
        "--out-root",
        default="data/SpaceNet3_prepared/GlobalScaleRoad",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples (0 = all).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    image_tar = Path(args.image_tar)
    geo_tar = Path(args.geojson_tar)
    out_root = Path(args.out_root)
    if not image_tar.exists():
        raise FileNotFoundError(f"Image tarball not found: {image_tar}")
    if not geo_tar.exists():
        raise FileNotFoundError(f"Geojson tarball not found: {geo_tar}")

    with tarfile.open(image_tar, "r:gz") as tf_img, tarfile.open(geo_tar, "r:gz") as tf_geo:
        img_members = _collect_members(tf_img, ".tif", IMAGE_ID_RE)
        geo_members = _collect_members(tf_geo, ".geojson", GEO_ID_RE)
        common_ids = sorted(set(img_members) & set(geo_members))
        if not common_ids:
            raise RuntimeError("No paired TIFF/GEOJSON files found in tarballs.")

        if int(args.limit) > 0:
            common_ids = common_ids[: int(args.limit)]

        split_map = _split_ids(common_ids, seed=int(args.seed), val_fraction=float(args.val_fraction), test_fraction=float(args.test_fraction))
        counts = {"train": 0, "val": 0, "in-domain-test": 0}

        for split in ("train", "val", "in-domain-test"):
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "masks").mkdir(parents=True, exist_ok=True)

        for idx, sample_id in enumerate(common_ids, start=1):
            split = split_map[sample_id]
            stem = f"spacenet3_paris_img{sample_id:04d}"
            image_out = out_root / split / "images" / f"{stem}.png"
            mask_out = out_root / split / "masks" / f"{stem}.png"
            if not args.force and image_out.exists() and mask_out.exists():
                counts[split] += 1
                continue

            img_bytes = tf_img.extractfile(img_members[sample_id]).read()  # type: ignore[union-attr]
            geo_bytes = tf_geo.extractfile(geo_members[sample_id]).read()  # type: ignore[union-attr]
            geo_payload = json.loads(geo_bytes.decode("utf-8"))

            with tifffile.TiffFile(io.BytesIO(img_bytes)) as tif:
                page = tif.pages[0]
                geotags = _extract_geotags(page)
                arr = tif.asarray()

            rgb = _to_rgb_uint8(arr)
            h, w = rgb.shape[:2]
            mask = _rasterize_mask(
                geo_payload,
                height=h,
                width=w,
                geotags=geotags,
                line_width=int(args.line_width),
            )

            Image.fromarray(rgb).save(image_out)
            Image.fromarray(mask).save(mask_out)
            counts[split] += 1

            if idx % 25 == 0 or idx == len(common_ids):
                print(f"[{idx}/{len(common_ids)}] prepared {stem} -> {split}", flush=True)

    summary = {
        "source_image_tar": str(image_tar),
        "source_geojson_tar": str(geo_tar),
        "out_root": str(out_root),
        "total_pairs": len(common_ids),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "line_width": int(args.line_width),
        "counts": counts,
    }
    summary_path = out_root / "spacenet3_paris_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
