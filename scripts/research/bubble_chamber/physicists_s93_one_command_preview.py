#!/usr/bin/env python3
"""One command: original private H50Z BMP -> source-only S92 -> small labeled JPEG.

Outputs a compact side-by-side preview and full-resolution detail crop sheet.
This does not upload medical or physics film data to GitHub or any service.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s93_compact_real_overlay import run_one_command


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmp",type=Path,required=True,
                        help="Original 8-bit grayscale BMP path (e.g. H50Z_030_01_405.bmp)")
    parser.add_argument("--private-out-dir",type=Path,required=True,
                        help="New private output directory OUTSIDE Git repository")
    parser.add_argument("--method",choices=("s88-steerable","frangi","contrast","azunet"),
                        default="s88-steerable",
                        help="Frozen source detector; s88 is a classical ridge detector, not trained ANZA")
    parser.add_argument("--checkpoint",type=Path,
                        help="Required for real original AZUNet; never claim ANZA otherwise")
    parser.add_argument("--max-side",type=int,default=2048,
                        help="Largest image dimension for condensed source/overlay JPEG (512..4096)")
    parser.add_argument("--max-gap-px",type=float,default=26.,
                        help="Advisory geometric join distance; not a physical track acceptance")
    parser.add_argument("--preview-limit",type=int,default=12,
                        help="Number of reviewed local high-resolution source+overlay crops")
    parser.add_argument("--resume",action="store_true",
                        help="Resume same verified S86 run; never overwrite completed S92 image evidence")
    args=parser.parse_args(argv)
    outcome=run_one_command(
        bmp_path=args.bmp,
        private_root=args.private_out_dir,
        method=args.method,
        checkpoint=args.checkpoint,
        max_side=args.max_side,
        max_gap_px=args.max_gap_px,
        preview_limit=args.preview_limit,
        resume=args.resume,
    )
    print(json.dumps(outcome,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
