#!/usr/bin/env python3
"""S100: autonomous source-locked 2D X-shaped crossing candidates.

This DOES NOT recognize a physical fiducial/cross vs two overlapping tracks
vs an interaction. It seeks four geometric arms WITHOUT operator f1..f8
locations, drawing only truly observed S86/S87 lines or SHA-validated S86
mask split arms as SOURCE observations; no invented source bridge.

The original H50Z BMP and all S86/S92/S95/S96 outputs are PRIVATE. Gallery
and tiny zip are diagnostic machine-selected examples, not GT/accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.s100_cross_geometry import XShapeConfig
from bubble_chamber.s100_cross_review import run_s100_unlabeled_cross_review


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bmp",type=Path,required=True,
                   help="Exact original uncompressed 8-bit grayscale H50Z.bmp, not resized JPG")
    p.add_argument("--s86-dir",type=Path,required=True)
    p.add_argument("--s92-dir",type=Path,required=True)
    p.add_argument("--s95-dir",type=Path,required=True)
    p.add_argument("--s96-dir",type=Path,required=True,
                   help="S96 run with --mask-junction-audit to also find fused pixel X branches")
    p.add_argument("--private-out-dir",type=Path,required=True)
    p.add_argument("--case",default="H50Z_REAL")
    p.add_argument("--max-preview-cases",type=int,default=12)
    p.add_argument("--crop-size",type=int,default=320)
    p.add_argument("--max-opposite-deflection-deg",type=float,default=30.)
    p.add_argument("--min-cross-angle-deg",type=float,default=35.)
    p.add_argument("--min-pairing-margin-deg",type=float,default=10.)
    p.add_argument("--max-center-to-port-px",type=float,default=14.)
    p.add_argument("--max-source-junction-records",type=int,default=100000)
    args=p.parse_args(argv)
    cfg=XShapeConfig(
        max_opposite_deflection_deg=args.max_opposite_deflection_deg,
        min_axes_cross_angle_deg=args.min_cross_angle_deg,
        min_best_vs_second_pairing_margin_deg=args.min_pairing_margin_deg,
        max_center_to_port_px=args.max_center_to_port_px,
        max_junction_records=args.max_source_junction_records,
    )
    result=run_s100_unlabeled_cross_review(
        bmp_path=args.bmp,
        s86_dir=args.s86_dir,s92_dir=args.s92_dir,
        s95_dir=args.s95_dir,s96_dir=args.s96_dir,
        private_out_dir=args.private_out_dir,
        case_name=args.case,max_review_cases=args.max_preview_cases,
        crop_size=args.crop_size,cfg=cfg
    )
    print(json.dumps({
        "status":result["status"],
        "source_BMP":result["source_bmp_name"],
        "source_BMP_SHA256":result["source_bmp_sha256"],
        "source_original_endpoint_junctions":result["original_S86_endpoint_junctions_scored"],
        "source_mask_derived_junctions":result["alternative_verified_S86_mask_junctions_scored"],
        "geometric_X_review_candidates_not_fiducials":result["four_arm_geometric_X_review_candidates"],
        "candidate_abstention_reasons":result["reason_counts"],
        "actual_photo_overlay_cases":result["actual_original_film_previews_exported"],
        "PRIVATE_image_gallery":str(args.private_out_dir/"index.html"),
        "PRIVATE_compact_zip":str(args.private_out_dir/result["compact_source_real_image_evidence_zip"]),
        "PRIVATE_compact_zip_sha256":result["compact_source_real_image_evidence_zip_sha256"],
        "not_fiducial_or_physical_particle_GT":True,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
