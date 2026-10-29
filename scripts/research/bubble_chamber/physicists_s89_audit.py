#!/usr/bin/env python3
"""PRIVATE, no-GT S89 image-space review of S88 full-BMP proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.s89_private_candidate_audit import AuditConfig, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s86-dir', type=Path, required=True)
    parser.add_argument('--bmp', type=Path, required=True)
    parser.add_argument('--private-out-dir', type=Path, required=True)
    parser.add_argument('--max-tiles', type=int, default=16)
    parser.add_argument('--preview-tiles', type=int, default=12)
    args = parser.parse_args(argv)
    report = run(args.s86_dir, args.bmp, args.private_out_dir,
                 AuditConfig(max_tiles=args.max_tiles, preview_tiles=args.preview_tiles))
    print(json.dumps({k: report[k] for k in (
        'status', 'sampled_tile_count', 'sampled_fragment_count',
        'review_first_fragment_count', 'physics_track_GT_count',
        'physical_auto_joins_authorized')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
