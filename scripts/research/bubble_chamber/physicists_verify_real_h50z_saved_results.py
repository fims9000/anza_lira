#!/usr/bin/env python3
"""Verify saved REAL H50Z S86/S92/S95/S96 machine source files, if available.

This is a read-only evidence verifier, NOT a processor that manufactures
"real" output. It refuses incomplete/mismatched stage sources; optional BMP
allows checking original bytes. Without stage files it FAILS, never inserts
previously asserted counts as if measured.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from bubble_chamber.real_bmp_run_receipts import (
    verify_stage_receipts,summarize_verification,html_summary
)


def _source(text):
    parts=text.split("|")
    if len(parts) not in (4,5,6) or any(not x.strip() for x in parts[:4]):
        raise ValueError(
            "Use --case ID|S86_DIR|S92_DIR|S95_DIR[|S96_DIR[|ORIGINAL_BMP]]; "
            "use '-' for an absent S96 stage"
        )
    name=parts[0]
    folder=[Path(x).resolve() for x in parts[1:4]]
    s96=(None if len(parts)<5 or parts[4]=="-" else Path(parts[4]).resolve())
    bmp=(None if len(parts)<6 or parts[5]=="-" else Path(parts[5]).resolve())
    return name,folder[0],folder[1],folder[2],s96,bmp


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",action="append",required=True,
                        help="ID|S86_DIR|S92_DIR|S95_DIR[|S96_DIR[|ORIGINAL_BMP]]; repeat for different photos")
    parser.add_argument("--private-out-dir",type=Path,required=True,
                        help="new PRIVATE destination containing verified JSON/HTML reports")
    args=parser.parse_args(argv)
    out=args.private_out_dir.resolve()
    root=ROOT.resolve()
    if out.exists() or out==root or root in out.parents:
        raise ValueError("Use a NEW private destination outside the repository")
    sources=[_source(x) for x in args.case]
    raw_paths=[p for row in sources for p in row[1:] if p is not None]
    if any(out==p or p in out.parents or out in p.parents for p in raw_paths):
        raise ValueError("Never put audit outputs within or above a source stage")
    verified=[
        verify_stage_receipts(
            case_name=name,s86_dir=s86,s92_dir=s92,s95_dir=s95,
            s96_dir=s96,original_bmp=bmp
        )
        for name,s86,s92,s95,s96,bmp in sources
    ]
    report=summarize_verification(verified)
    out.mkdir(mode=0o700,parents=True)
    out.chmod(0o700)
    file=out/"source_verified_machine_counts.json"
    with file.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,sort_keys=True)
        stream.write("\n")
    file.chmod(0o600)
    page=out/"index.html"
    with page.open("x",encoding="utf-8") as stream:
        stream.write(html_summary(report))
    page.chmod(0o600)
    print(json.dumps({
        "status":report["status"],
        "verified_cases":report["verified_case_count"],
        "real_BMP_byte_verified_cases":report["real_bmp_bytes_verified_case_count"],
        "summary":str(file),
        "HTML":str(page),
        "machine_only_not_physical_ground_truth":True,
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
