#!/usr/bin/env python3
"""S95 expert visual audit of source-locked gap proposals (not particle truth).

pack: create offline blinded original-vs-geometry crops from frozen S86/S92/S95.
compile: verify manually exported responses from this exact pack, report ONLY
       appearance-limited rates among expressly reviewed examples.
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
from bubble_chamber.s87_seam_continuity import _read_source
from bubble_chamber.s92_curve_review import collapse_observed_components
from bubble_chamber.s95_gap_visual_review import (
    build_visual_gap_pack,compile_visual_gap_review,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("pack")
    p.add_argument("--s86-dir",type=Path,required=True)
    p.add_argument("--s92-dir",type=Path,required=True)
    p.add_argument("--s95-dir",type=Path,required=True)
    p.add_argument("--bmp",type=Path)
    p.add_argument("--private-out-dir",type=Path,required=True)
    p.add_argument("--max-cases",type=int,default=48)
    p.add_argument("--crop-size",type=int,default=320)
    q=sub.add_parser("compile")
    q.add_argument("--pack-dir",type=Path,required=True)
    q.add_argument("--review-json",type=Path,required=True)
    q.add_argument("--private-out-json",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=="pack":
        manifest,records,manifest_sha=_read_source(args.s86_dir)
        source=json.loads((args.s95_dir/"report.json").read_text(encoding="utf8"))
        if source.get("source_s86_manifest_sha256")!=manifest_sha:
            raise ValueError("S95 derived from another S86 original source")
        components,_=collapse_observed_components(records)
        result=build_visual_gap_pack(
            s95_dir=args.s95_dir,s92_dir=args.s92_dir,
            components=components,bmp_path=args.bmp,
            out_dir=args.private_out_dir,
            max_cases=args.max_cases,crop_size=args.crop_size,
        )
        print(json.dumps({
            "pack_status":result["status"],
            "cases":result["sampled_cases"],
            "original_photo_available":result["source_image_available"],
            "review_page":str(args.private_out_dir/"index.html"),
            "source_s92_proposals":result["all_machine_gap_proposals"],
        },ensure_ascii=False,indent=2))
    else:
        out=args.private_out_json.resolve()
        repo=ROOT.resolve()
        pack_dir=args.pack_dir.resolve()
        if out==repo or repo in out.parents or out.exists() or pack_dir in out.parents:
            raise ValueError("Use new private result outside Git and review pack")
        packfile=pack_dir/"review_pack.json"
        pack=json.loads(packfile.read_text(encoding="utf8"))
        human_file=args.review_json.resolve()
        source_sha=sha256_file(human_file)
        pack_sha=sha256_file(packfile)
        result=compile_visual_gap_review(
            pack,json.loads(human_file.read_text(encoding="utf-8-sig"))
        )
        if (sha256_file(human_file)!=source_sha
            or sha256_file(packfile)!=pack_sha):
            raise ValueError("Visual reviewer input changed during verification")
        result["source_human_response_sha256"]=source_sha
        result["source_review_pack_sha256"]=pack_sha
        out.parent.mkdir(parents=True,exist_ok=True)
        out.parent.chmod(0o700)
        with out.open("x",encoding="utf8") as file:
            json.dump(result,file,ensure_ascii=False,indent=2,sort_keys=True)
            file.write("\n")
        out.chmod(0o600)
        print(json.dumps({
            "status":result["status"],
            "reviewed":result["reviewed_cases"],
            "unreviewed_remain_unknown":result["unreviewed_cases_remain_unknown"],
            "reviewed_selected":result["selected_machine_proposals_reviewed"],
            "selected_apparent_false_join_fraction_among_decided":
                result["selected_apparent_false_join_fraction_among_decided"],
        },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
