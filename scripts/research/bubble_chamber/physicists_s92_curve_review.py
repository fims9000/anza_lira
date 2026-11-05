#!/usr/bin/env python3
"""Private S92 curve review: short ticks, hairpins, coiling, dashed track gaps.

All results remain UNKNOWN machine suggestions. Source S86 masks and BMP stay
unchanged; output is a new private local folder outside this Git repository.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s92_curve_review import run_s92
from bubble_chamber.s92_curve_geometry import TraceReviewConfig


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s86-dir",type=Path,required=True)
    parser.add_argument("--bmp",type=Path,help="frozen original 8-bit BMP; optional for private previews")
    parser.add_argument("--private-out-dir",type=Path,required=True)
    parser.add_argument("--preview-limit",type=int,default=20)
    parser.add_argument("--priority-centerline-tiles",action="store_true",
                        help="write display-only observed centerline PNG tiles, never change original masks")
    parser.add_argument("--max-gap",type=float,default=26.0,
                        help="max gap pixels for image-only candidate proposals (not physical joins)")
    args=parser.parse_args(argv)
    base=TraceReviewConfig()
    cfg=TraceReviewConfig(**{**vars(base),"max_gap_px":args.max_gap})
    report=run_s92(args.s86_dir,args.private_out_dir,bmp_path=args.bmp,
                   cfg=cfg,preview_limit=args.preview_limit,
                   export_priority_tiles=args.priority_centerline_tiles)
    print(json.dumps({
        k:report[k] for k in (
            "status","source_fragments","observed_components_after_S87",
            "shape_review_categories","gap","physical_auto_joins_authorized",
        )
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
