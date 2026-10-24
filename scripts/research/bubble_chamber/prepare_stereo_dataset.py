#!/usr/bin/env python3
"""Build a run-disjoint 3-view photo inventory; no stereo calibration invented."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bubble_chamber.stereo_dataset import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.input_csv, args.image_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({'events':manifest['event_count'], 'run_groups':manifest['run_group_count'], 'split':manifest['split_status'], 'out':str(args.out)}))

if __name__ == '__main__': main()
