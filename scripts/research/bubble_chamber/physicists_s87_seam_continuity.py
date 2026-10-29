#!/usr/bin/env python3
"""Safe native-pixel seam tracing after private S86 full-BMP segmentation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bubble_chamber.s87_seam_continuity import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s86-dir', required=True, type=Path)
    parser.add_argument('--private-out-dir', required=True, type=Path)
    parser.add_argument('--max-tangent-angle', type=float, default=55.0)
    parser.add_argument('--bmp', type=Path, help='Original matching PRIVATE BMP for gallery')
    parser.add_argument('--preview-limit', type=int, default=12)
    args = parser.parse_args(argv)
    result = run(args.s86_dir, args.private_out_dir, tangent_angle_deg=args.max_tangent_angle,
                 bmp_path=args.bmp, preview_limit=args.preview_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
