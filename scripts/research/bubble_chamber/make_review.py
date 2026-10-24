#!/usr/bin/env python3
"""Generate standalone local bubble-track review UI from prior P0 output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.review import make_review_html, validate_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True, type=Path)
    p.add_argument('--image', type=Path, help='Original photograph. Required for generating HTML.')
    p.add_argument('--out', type=Path, help='Output self-contained HTML')
    p.add_argument('--labels', type=Path, help='Validate an expert-labeled JSON file against this result')
    args = p.parse_args()
    result = json.loads(args.result.read_text(encoding='utf-8'))
    if args.labels:
        counts = validate_labels(result, json.loads(args.labels.read_text(encoding='utf-8')))
        print(json.dumps(counts, ensure_ascii=False))
        return
    if args.image is None or args.out is None:
        p.error('--image and --out are required for generating HTML')
    frame = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        p.error(f'Cannot read {args.image}')
    html = make_review_html(result, frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding='utf-8')
    print(f'HTML: {args.out} | candidates: {len(result["join_proposals"])} | bytes: {len(html):,}')


if __name__ == '__main__':
    main()
