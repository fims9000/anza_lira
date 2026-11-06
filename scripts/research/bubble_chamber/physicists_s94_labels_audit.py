#!/usr/bin/env python3
"""Audit original H50Z labels.json without photos; no 'track GT' invented.

Counts starts and fiducial marks, quarantines duplicate f1..f8 per photo,
checks every point's original pixel bounds. Output is a PRIVATE source-provenance
receipt, not any repaired ground-truth annotation file.
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
from bubble_chamber.s94_sparse_keypoint_guard import audit_label_collection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels",type=Path,required=True)
    parser.add_argument("--private-out-json",type=Path,required=True,
                        help="New private JSON report OUTSIDE source repository")
    args=parser.parse_args(argv)
    source=args.labels.resolve()
    dest=args.private_out_json.resolve()
    repo=ROOT.resolve()
    if dest==repo or repo in dest.parents:
        raise ValueError("No private data inside Git repository")
    if dest==source or dest.exists():
        raise FileExistsError("Cannot replace labels or existing audit")
    if not source.is_file():
        raise FileNotFoundError(source)
    checksum=sha256_file(source)
    labels=json.loads(source.read_text(encoding="utf8-sig"))
    result=audit_label_collection(labels)
    result["labels_sha256"]=checksum
    result["source_name"]=source.name
    result["original_labels_left_unchanged"]=True
    if sha256_file(source)!=checksum:
        raise ValueError("Original labels.json changed during audit")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.parent.chmod(0o700)
    with dest.open("x",encoding="utf8") as file:
        json.dump(result,file,ensure_ascii=False,sort_keys=True,indent=2)
        file.write("\n")
    dest.chmod(0o600)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
