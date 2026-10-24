#!/usr/bin/env python3
"""Extract ONLY a provided real photo from the source archive, preserving lineage.

No pseudo-labels, no random train/test splits of one film frame, no license claim.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ROIS = {
    "upper_left": (90, 75, 445, 560),
    "upper_right": (440, 80, 860, 640),
    "lower_left": (115, 630, 500, 1320),
    "lower_right": (470, 640, 840, 1370),
}


def prepare_archive(archive_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if item.filename.lower().endswith("/action.jpg")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one action.jpg, found {len(members)}")
        payload = archive.read(members[0])
    array = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if array is None or array.ndim != 2:
        raise ValueError("Failed to decode photo as grayscale image")
    h, w = array.shape
    source = out_dir / "action.jpg"
    source.write_bytes(payload)
    crops = []
    for name, (x0, y0, x1, y1) in ROIS.items():
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"ROI {name} does not fit frame {w}x{h}")
        path = out_dir / f"crop_{name}.png"
        cv2.imwrite(str(path), array[y0:y1, x0:x1])
        crops.append({"file": path.name, "roi_xyxy_in_original": [x0, y0, x1, y1], "parent": "action.jpg",
                      "sha256": sha256(path.read_bytes()).hexdigest(), "independent_sample": False})
    meta = {
        "record_id": "user_archive_action_reference_001",
        "origin": "user-provided archive: пузыри полезное.zip / старые автоматы / action.jpg",
        "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "image_sha256": sha256(payload).hexdigest(),
        "shape_yx": [h, w],
        "label_status": "no expert masks or track IDs",
        "camera_medium": "unverified",
        "film_id_run_id": "unverified",
        "stereo_views_available": False,
        "license_and_redistribution": "not determined; do not commit/circulate publicly without permission",
        "samples": crops,
        "split_policy": "All crops share the same parent frame; never split them across train and test.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    meta = prepare_archive(args.archive, args.out_dir)
    print(json.dumps({"image_sha256": meta["image_sha256"], "shape": meta["shape_yx"], "crops": len(meta["samples"]), "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
