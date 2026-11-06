#!/usr/bin/env python3
"""Single-command local REAL H50Z rerun + split compact evidence ZIP.

run: original 8-bit BMP -> verified S86/S92/S95 -> optional S96 ->
     validate all source SHA and create two compact independent zip packages.
pack: saved exact-origin S86/S92/S95(/S96) + BMP -> same zip packages.
score: after a blinded human manually completes the ORIGINAL-ONLY CSV,
       compile source-locked descriptive selected-candidate error report.

Never uploads original >200MB BMP to GitHub, never claims native physical GT,
never silently resumes a modified source/model/parameter set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s93_compact_real_overlay import run_one_command
from bubble_chamber.s95_all_structures_review import run_s95
from bubble_chamber.s96_junction_workflow import run_s96
from bubble_chamber.real_h50z_compact_evidence import (
    package_verified_real_h50z,evaluate_blind_component_cases,
    verify_existing_sealed_evidence,
)
from bubble_chamber.s100_cross_review import (
    run_s100_unlabeled_cross_review,verify_existing_s100_gallery
)


def _package(*,args,s86,s92,s95,s96,out_dir):
    if out_dir.exists() and getattr(args,"resume",False):
        report=verify_existing_sealed_evidence(
            source_bmp_path=args.bmp,s86_dir=s86,s92_dir=s92,
            s95_dir=s95,s96_dir=s96,evidence_dir=out_dir,
            case_id=args.case,max_per_class=args.max_per_class,
            crop_size=args.crop_size
        )
    else:
        report=package_verified_real_h50z(
        bmp_path=args.bmp,
        s86_dir=s86,
        s92_dir=s92,
        s95_dir=s95,
        s96_dir=s96,
        private_out_dir=out_dir,
        max_per_class=args.max_per_class,
        crop_size=args.crop_size,
        case_id=args.case,
        )
    print(json.dumps({
        "status":report["status"],
        "source_bmp_sha256":report["source_bmp_sha256"],
        "original_full_BMP_not_in_zip":True,
        "source_verified_original_components":report["original_verified_S86_S95_count"],
        "examples":report["blind_source_examples"],
        "first_open_blind_zip":str(out_dir/report["archives"]["blind"]["file"]),
        "only_after_blind_review_machine_zip":str(out_dir/report["archives"]["posthoc_machine"]["file"]),
        "compressed_zip_sha256s":{
            k:v["sha256"] for k,v in report["archives"].items()
        },
        "not_exhaustive_GT":True,
    },ensure_ascii=False,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    cmds=p.add_subparsers(dest="command",required=True)
    run=cmds.add_parser("run",help="Full H50Z source-bmp S88/actual AZUNet run and compact QA deliverables")
    run.add_argument("--bmp",type=Path,required=True)
    run.add_argument("--work-root",type=Path,required=True,
                     help="PRIVATE new root for all intermediate genuine S86/S92/S95(/S96) output")
    run.add_argument("--method",choices=["s88-steerable","frangi","contrast","azunet"],
                     default="s88-steerable")
    run.add_argument("--checkpoint",type=Path,help="Only for actual trained AZUNet")
    run.add_argument("--labels",type=Path,
                     help="Optional exact-photo matched sparse H50Z operator labels.json")
    run.add_argument("--operator-curves",type=Path,
                     help="Only exact-photo SHA matched operator source curves")
    run.add_argument("--resume",action="store_true",
                     help="Resume only verified same-origin S93 source runs, never overwrite completed stages")
    run.add_argument("--skip-s96",action="store_true",
                     help="S96 optional, omit if only morphology diagnosis is needed")
    run.add_argument("--mask-junction-audit",action="store_true",
                     help="Extra CPU original S86 binary skeleton 4-arm audit")
    run.add_argument("--s100-cross-review",action="store_true",
                     help="Also export S100 original-BMP geometric X-shapes without operator f1..f8")
    run.add_argument("--s100-review-limit",type=int,default=12,
                     help="Max native-source geometric X visual examples (0..48)")
    run.add_argument("--max-fragments-per-tile",type=int,default=2048)
    run.add_argument("--min-segment-length-px",type=float,default=4.)
    run.add_argument("--min-component-area",type=int,default=2)
    run.add_argument("--max-side",type=int,default=2048)
    run.add_argument("--max-per-class",type=int,default=2)
    run.add_argument("--crop-size",type=int,default=384)
    run.add_argument("--case",default="H50Z_REAL")
    pack=cmds.add_parser("pack",help="Already processed source stage directories, WITHOUT running any segmentation")
    pack.add_argument("--bmp",type=Path,required=True)
    pack.add_argument("--s86-dir",type=Path,required=True)
    pack.add_argument("--s92-dir",type=Path,required=True)
    pack.add_argument("--s95-dir",type=Path,required=True)
    pack.add_argument("--s96-dir",type=Path)
    pack.add_argument("--private-out-dir",type=Path,required=True)
    pack.add_argument("--max-per-class",type=int,default=2)
    pack.add_argument("--crop-size",type=int,default=384)
    pack.add_argument("--case",default="H50Z_REAL")
    score=cmds.add_parser("score",help="After expert completed BLIND original-only CSV, descriptive candidate agreement")
    score.add_argument("--machine-report-json",type=Path,required=True)
    score.add_argument("--completed-blind-csv",type=Path,required=True)
    score.add_argument("--private-out-json",type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=="score":
        result=evaluate_blind_component_cases(
            machine_report_path=args.machine_report_json,
            blind_review_csv=args.completed_blind_csv,
            private_out_json=args.private_out_json,
        )
        print(json.dumps({
            "status":result["status"],
            "machine_selected_examples":result["sampled_machine_components"],
            "expert_clear_consistent":result["visual_consistent_cases"],
            "expert_clear_contradictory":result["visual_contradictory_cases"],
            "uncertain_abstentions":result["human_uncertain_cases_abstained"],
            "descriptive_consistency_not_population_precision":
                result["descriptive_consistency_among_human_clear_selected_candidates"],
            "not_full_physics_GT":True,
        },ensure_ascii=False,indent=2))
        return
    if args.command=="pack":
        _package(args=args,s86=args.s86_dir,s92=args.s92_dir,
                 s95=args.s95_dir,s96=args.s96_dir,
                 out_dir=args.private_out_dir.resolve())
        return
    root=args.work_root.resolve()
    repo=ROOT.resolve()
    if root==repo or repo in root.parents:
        raise ValueError("S86-S96 real run outputs must remain PRIVATE outside Git")
    if root.exists() and not args.resume:
        raise ValueError("H50Z work root already exists; use --resume after confirming same photo or new directory")
    if args.method=="azunet" and args.checkpoint is None:
        raise ValueError("H50Z AZUNet requires an actual checkpoint, not a renamed S88 baseline")
    if args.method!="azunet" and args.checkpoint is not None:
        raise ValueError("Checkpoint can only accompany actual trained AZUNet")
    if args.mask_junction_audit and args.skip_s96:
        raise ValueError("Mask-junction audit requires actual S96 stage")
    if args.s100_cross_review and args.skip_s96:
        raise ValueError("S100 unlabeled source-X review requires S96 stage")
    if args.s100_cross_review and not 0<=args.s100_review_limit<=48:
        raise ValueError("S100 source photo preview budget must be 0..48")
    # S93 handles strict SHA+params identity and fails closed when resuming.
    s93=root/"S93_SOURCES"
    run_one_command(
        args.bmp,s93,method=args.method,checkpoint=args.checkpoint,
        max_side=args.max_side,labels_path=args.labels,
        resume=args.resume,
        min_segment_length_px=args.min_segment_length_px,
        min_component_area=args.min_component_area,
        max_fragments_per_tile=args.max_fragments_per_tile
    )
    s86=s93/"S86"
    s92=s93/"S92"
    s95=root/"S95_ALL_OBJECTS"
    if not s95.exists():
        run_s95(
            s86,s92,s95,bmp_path=args.bmp,labels_path=args.labels,
            operator_curves_path=args.operator_curves,max_side=args.max_side
        )
    elif not args.resume:
        raise ValueError("Cannot overwrite existing S95 result")
    s96=None
    if not args.skip_s96:
        s96=root/"S96_JUNCTIONS"
        if not s96.exists():
            run_s96(
                s86,s95,s96,bmp_path=args.bmp,preview_limit=16,
                mask_junction_audit=args.mask_junction_audit
            )
        elif not args.resume:
            raise ValueError("Cannot overwrite existing S96 result")
    # Only after all stages (including exact real BMP) pass receipt checks
    # does the packer create a compact result. It never regenerates "counts"
    # from previously pasted conversational numbers.
    _package(args=args,s86=s86,s92=s92,s95=s95,s96=s96,
             out_dir=root/"COMPACT_REAL_EVIDENCE")
    if args.s100_cross_review:
        crosses=root/"S100_UNLABELED_X_CROSS_SHAPES"
        if crosses.exists() and args.resume:
            cross_result=verify_existing_s100_gallery(
                bmp_path=args.bmp,s86_dir=s86,s92_dir=s92,
                s95_dir=s95,s96_dir=s96,out_dir=crosses,
                max_review_cases=args.s100_review_limit,
                crop_size=min(512,max(160,args.crop_size))
            )
        else:
            cross_result=run_s100_unlabeled_cross_review(
                bmp_path=args.bmp,s86_dir=s86,s92_dir=s92,
                s95_dir=s95,s96_dir=s96,private_out_dir=crosses,
                case_name=args.case,
                max_review_cases=args.s100_review_limit,
                crop_size=min(512,max(160,args.crop_size))
            )
        print(json.dumps({
            "S100_status":cross_result["status"],
            "source_only_four_arm_geometric_X_count_not_true_fiducials":
                cross_result["four_arm_geometric_X_review_candidates"],
            "source_mask_virtual_arm_evidence_available":
                cross_result["alternative_verified_S86_mask_junctions_scored"]>0,
            "original_photo_posthoc_gallery":str(crosses/"index.html"),
            "source_pixel_exact_cross_examples_zip":str(crosses/
                cross_result["compact_source_real_image_evidence_zip"]),
            "no_physical_particle_identity_established":True,
        },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
