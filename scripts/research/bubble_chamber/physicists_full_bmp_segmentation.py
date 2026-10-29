#!/usr/bin/env python3
"""Private full-resolution BMP segmentation; machine proposals, never physics GT."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bmp', required=True, type=Path)
    parser.add_argument('--private-out-dir', required=True, type=Path)
    parser.add_argument('--method', choices=('contrast', 'frangi', 'azunet'), default='frangi')
    parser.add_argument('--checkpoint', type=Path, help='trusted source-locked AZUNet .pt (PRIVATE)')
    parser.add_argument('--core', type=int, default=384)
    parser.add_argument('--halo', type=int, default=48)
    parser.add_argument('--min-length', type=float, default=12.0)
    parser.add_argument('--preview-limit', type=int, default=16)
    parser.add_argument('--full-mask-npy', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    cfg = FullBMPConfig(method=args.method, core_size=args.core, halo=args.halo,
                        min_segment_length_px=args.min_length,
                        preview_limit=args.preview_limit, full_mask_npy=args.full_mask_npy)
    result = run_full_bmp(args.bmp, args.private_out_dir, cfg,
                          checkpoint=args.checkpoint, resume=args.resume)
    print(json.dumps({k: result[k] for k in (
        'status', 'tile_count', 'candidate_fragments', 'candidate_foreground_pixels',
        'seam_advisories_UNKNOWN', 'tiles_with_fragment_extraction_skipped',
        'physical_auto_joins_authorized')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
