#!/usr/bin/env python3
"""S95: classify every S86 source fragment as a reviewable morphology type.

Works directly with existing S86+S92 output WITHOUT the 207MB BMP.
With --bmp alone, runs the full S93 -> S95 workflow and exports a small
real-photo overlay showing all objects (no deliberate clutter hiding).
Never pretends track/coil/cross is a known particle or physics truth.
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
from bubble_chamber.s95_all_structures_review import run_s95


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmp",type=Path,
                        help="Original source-locked BMP; omit to classify existing S86/S92 without image")
    parser.add_argument("--s86-dir",type=Path,
                        help="Existing complete private S86 folder (requires --s92-dir)")
    parser.add_argument("--s92-dir",type=Path,
                        help="Existing complete private S92 folder (requires --s86-dir)")
    parser.add_argument("--labels",type=Path,
                        help="Optional exact-photo H50Z start/f1..f8 labels.json")
    parser.add_argument("--operator-curves",type=Path,
                        help="Optional manual SOURCE-SHA matched 2D quadratic/measurement guide JSON")
    parser.add_argument("--private-out-dir",type=Path,required=True,
                        help="New private output outside Git and source directories")
    parser.add_argument("--method",choices=("s88-steerable","frangi","contrast","azunet"),
                        default="s88-steerable")
    parser.add_argument("--checkpoint",type=Path,
                        help="For original source-locked AZUNet only")
    parser.add_argument("--max-side",type=int,default=2048)
    parser.add_argument("--min-length-px",type=float,default=4.,
                        help="High-recall S86 extraction includes >=4px strokes; much noisier than S93 default")
    parser.add_argument("--min-component-area",type=int,default=2,
                        help="Keep small original foreground components as candidates")
    parser.add_argument("--max-fragments-per-tile",type=int,default=2048,
                        help="Increase bounded extraction budget; inspect skipped S86 tiles")
    parser.add_argument("--resume-s93",action="store_true",
                        help="If S93 partially ran previously, safely resume it")
    args=parser.parse_args(argv)

    if bool(args.s86_dir)!=bool(args.s92_dir):
        parser.error("--s86-dir and --s92-dir are required together")
    if args.s86_dir is None and args.bmp is None:
        parser.error("Provide either --bmp or --s86-dir with --s92-dir")

    out=args.private_out_dir.resolve()
    if args.s86_dir is None:
        run_one_command(
            args.bmp,out/"S93_SOURCES",method=args.method,
            checkpoint=args.checkpoint,labels_path=args.labels,
            max_side=args.max_side,resume=args.resume_s93,
            min_segment_length_px=args.min_length_px,
            min_component_area=args.min_component_area,
            max_fragments_per_tile=args.max_fragments_per_tile,
        )
        s86=out/"S93_SOURCES"/"S86"
        s92=out/"S93_SOURCES"/"S92"
    else:
        s86=args.s86_dir
        s92=args.s92_dir
    result=run_s95(
        s86,s92,out/"S95_ALL_OBJECTS",
        bmp_path=args.bmp,
        labels_path=args.labels,
        operator_curves_path=args.operator_curves,
        max_side=args.max_side,
    )
    print(json.dumps({
        "status":result["status"],
        "source_photo":result["source_bmp_name"],
        "class_counts":result["all_extracted_line_categories"]["categories"],
        "all_components_preserved":result["source_fragments_deleted"]==0,
        "s86_tiles_missing_fragment_extraction":result["source_s86_tiles_skipped"],
        "photo_overlay":(
            str(out/"S95_ALL_OBJECTS"/result["all_classes_comparison_file"])
            if result["all_classes_comparison_file"] else None
        ),
        "private_report":str(out/"S95_ALL_OBJECTS"/"report.json"),
        "not_expert_physics_GT":result["not_ground_truth_segmentation"],
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
