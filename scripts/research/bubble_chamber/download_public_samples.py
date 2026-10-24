#!/usr/bin/env python3
"""Fetch a tiny public-domain *unlabeled* bubble-chamber image set.

Two independent archival records, not detector-matched training/test events.
No network requests during import. Images are never committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

SOURCES = (
    {
        'id': 'usdoe_hydrogen_1955',
        'filename': 'usdoe_hydrogen_1955.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/7/76/Liquid_hydrogen_bubblechamber.jpg',
        'source_page': 'https://commons.wikimedia.org/wiki/File:Liquid_hydrogen_bubblechamber.jpg',
        'rights': 'public_domain_us_government_work_per_source_page',
        'medium': 'liquid_hydrogen',
    },
    {
        'id': 'usdoe_hydrogen_event',
        'filename': 'usdoe_hydrogen_event.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/1/14/HD.6D.635_%2812000265314%29.jpg',
        'source_page': 'https://commons.wikimedia.org/wiki/File:HD.6D.635_(12000265314).jpg',
        'rights': 'public_domain_us_government_work_per_source_page',
        'medium': 'liquid_hydrogen',
    },
)


def _save_sample(src: dict, output: Path, data: bytes) -> dict:
    if len(data) > 15_000_000:
        raise ValueError('unexpectedly large image')
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None or arr.ndim != 2 or min(arr.shape) < 100:
        raise ValueError(f"not a plausible archival photograph: {src['id']}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / src['filename']
    target.write_bytes(data)
    return {
        **src,
        'sha256_file': hashlib.sha256(data).hexdigest(),
        'shape_yx': list(arr.shape),
        'label_status': 'NO_TRACK_LABELS',
        'stereo_triplet': False,
        'calibration': 'not_provided',
        'experiment_role': 'qualitative_domain_probe_only',
    }


def fetch_subset(output: Path, sources=SOURCES) -> dict:
    entries = []
    for src in sources:
        request = Request(src['url'], headers={'User-Agent': 'bubble-chamber-research/0.1 (educational, small archival sample)'})
        with urlopen(request, timeout=15) as response:
            raw = response.read(15_000_001)
        entries.append(_save_sample(src, output, raw))
    meta = {
        'dataset': 'us_doe_bubble_chamber_unlabeled_miniset',
        'not_ground_truth': True,
        'claim': 'Photographs for smoke testing only. No true particle IDs, segmentation masks, or calibration.',
        'entries': entries,
    }
    (output / 'public_manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    manifest = fetch_subset(args.out_dir)
    print(json.dumps({'saved': len(manifest['entries']), 'out_dir': str(args.out_dir)}))


if __name__ == '__main__':
    main()
