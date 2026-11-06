#!/usr/bin/env python3
"""Validate S95 offline human review decisions, preserve an immutable receipt.

Manual categories alone do not verify particle species, physics, or correct
same-track gap joins. No source S86/S92/S95 files or photos are changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.bitmap_tiles import sha256_file
from bubble_chamber.s95_human_review import compile_human_review


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s95-dir",required=True,type=Path)
    parser.add_argument("--human-review-json",required=True,type=Path)
    parser.add_argument("--private-out-json",required=True,type=Path)
    args=parser.parse_args(argv)

    s95=args.s95_dir.resolve()
    input_file=args.human_review_json.resolve()
    out=args.private_out_json.resolve()
    if out==ROOT or ROOT in out.parents:
        raise ValueError("Human review output must stay outside source Git")
    if out==input_file or out.exists():
        raise FileExistsError("Cannot overwrite an existing human review or receipt")
    if s95==out or s95 in out.parents:
        raise ValueError("Do not overwrite machine S95 source")
    report_file=s95/"report.json"
    source_file=s95/"all_source_fragment_types.ndjson"
    report=json.loads(report_file.read_text(encoding="utf8"))
    objects=[json.loads(line)
             for line in source_file.read_text(encoding="utf8").splitlines()
             if line.strip()]
    input_sha=sha256_file(input_file)
    machine_sha=sha256_file(report_file)
    human=json.loads(input_file.read_text(encoding="utf-8-sig"))
    result=compile_human_review(human,report,machine_sha,objects)
    result["human_review_source_sha256"]=input_sha
    result["machine_S95_class_list_sha256"]=sha256_file(source_file)
    if (sha256_file(input_file)!=input_sha
        or sha256_file(report_file)!=machine_sha):
        raise ValueError("Review or source changed during validation")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.parent.chmod(0o700)
    with out.open("x",encoding="utf8") as file:
        json.dump(result,file,ensure_ascii=False,sort_keys=True,indent=2)
        file.write("\n")
    out.chmod(0o600)
    print(json.dumps({
        "status":result["status"],
        "all_source_objects":result["all_source_objects"],
        "human_reviewed":result["human_reviewed"],
        "unreviewed_remain_unknown":result["unreviewed_remain_unknown"],
        "human_status_counts":result["human_status_counts"],
        "human_morphology_counts":result["human_morphology_counts"],
        "private_out_json":str(out),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
