#!/usr/bin/env python3
"""S96 full observed-source branch review and independent crossing adjudication.

run: S86/S95 -> crossing alternatives -> SOURCE-LOCKED original BMP blind UI.
compile: verify exported operator 2D pair choices and report partial agreement.
No original image needed for read-only geometry, but human answers REQUIRE BMP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s96_junction_graph import JunctionConfig
from bubble_chamber.s96_junction_workflow import run_s96
from bubble_chamber.s96_junction_human_review import compile_junction_review


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="action",required=True)
    r=commands.add_parser("run",help="source-locked junction alternatives + blind crop viewer")
    r.add_argument("--s86-dir",type=Path,required=True)
    r.add_argument("--s95-dir",type=Path,required=True)
    r.add_argument("--bmp",type=Path,help="exact original BMP for expert photo review")
    r.add_argument("--private-out-dir",type=Path,required=True)
    r.add_argument("--preview-limit",type=int,default=32)
    r.add_argument("--crop-size",type=int,default=320)
    r.add_argument("--junction-radius",type=float,default=12.)
    r.add_argument("--min-rival-margin",type=float,default=.12)
    c=commands.add_parser("compile",help="compile human choices without physical-truth claims")
    c.add_argument("--review-pack",type=Path,required=True)
    c.add_argument("--human-review-json",type=Path,required=True)
    c.add_argument("--private-out-json",type=Path,required=True)
    args=p.parse_args(argv)
    if args.action=="run":
        cfg=JunctionConfig(
            max_endpoint_center_radius_px=args.junction_radius,
            min_rival_score_margin=args.min_rival_margin,
        )
        out=run_s96(args.s86_dir,args.s95_dir,args.private_out_dir,
                    bmp_path=args.bmp,cfg=cfg,
                    preview_limit=args.preview_limit,
                    crop_size=args.crop_size)
        print(json.dumps({
            "status":out["status"],
            "source_photo":out["source_bmp_name"],
            "source_components_preserved":out["geometry"]["source_components_preserved"],
            "junctions":out["geometry"].get("candidate_junctions",0),
            "provisional_2d_pairs":out["geometry"].get("provisional_visual_continuation_pairs",0),
            "review_cases":out["blinded_review_cases"],
            "source_photo_available":out["review_image_available"],
            "private_review_page":out["private_blinded_review_page"],
            "not_physical_track_GT":out["true_particles_cannot_be_inferred_from_2D_alone"],
        },ensure_ascii=False,indent=2))
    else:
        out=args.private_out_json.resolve()
        source=args.human_review_json.resolve()
        packfile=args.review_pack.resolve()
        repo=ROOT.resolve()
        if (out.exists() or out==source or out==packfile or
            out==repo or repo in out.parents or packfile.parent in out.parents):
            raise ValueError("Output must be a NEW private receipt outside source and Git")
        pack_sha=sha256_file(packfile)
        response_sha=sha256_file(source)
        pack=json.loads(packfile.read_text(encoding="utf-8"))
        reviewer=json.loads(source.read_text(encoding="utf-8-sig"))
        summary=compile_junction_review(pack,reviewer)
        if sha256_file(packfile)!=pack_sha or sha256_file(source)!=response_sha:
            raise ValueError("Edited source package during review validation")
        summary["source_review_pack_sha256"]=pack_sha
        summary["source_human_review_sha256"]=response_sha
        out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        out.parent.chmod(0o700)
        with out.open("x",encoding="utf-8") as stream:
            json.dump(summary,stream,ensure_ascii=False,sort_keys=True,indent=2)
            stream.write("\n")
        out.chmod(0o600)
        print(json.dumps({
            "status":summary["status"],
            "reviewed_cases":summary["reviewed_cases"],
            "unreviewed_cases_unknown":summary["unreviewed_cases_unknown"],
            "human_visual_agreement_rate":summary["machine_human_visual_pairing_agreement_rate"],
            "not_physical_truth":summary["not_physical_track_or_collision_accuracy"],
            "private_receipt":str(out),
        },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
