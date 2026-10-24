#!/usr/bin/env python3
"""Prepare tiny lossless annotated BMP crops without shipping whole 200 MB films."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.bitmap_tiles import export_h50z_tiles, export_visualmetro_triplet_tiles
from bubble_chamber.visualmetro_import import parse_track


def _archive(root: Path, zip_path: Path) -> None:
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(root.glob("*.png")):
            z.write(p, p.name)
        z.write(root / "manifest.json", "manifest.json")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    h=sub.add_parser("h50z",help="crop only the locally present BMPs matching labels.json")
    h.add_argument("--image-dir",type=Path,required=True)
    h.add_argument("--labels",type=Path,required=True)
    h.add_argument("--out-dir",type=Path,required=True)
    h.add_argument("--tile-size",type=int,default=1024)
    h.add_argument("--no-fiducials",action="store_true")
    v=sub.add_parser("visualmetro",help="crop matched point indices in three *explicit* film views")
    v.add_argument("--track-file",type=Path,required=True)
    v.add_argument("--view1",type=Path,required=True)
    v.add_argument("--view2",type=Path,required=True)
    v.add_argument("--view3",type=Path,required=True)
    v.add_argument("--out-dir",type=Path,required=True)
    v.add_argument("--indices",type=str,default="0,middle,last")
    v.add_argument("--tile-size",type=int,default=1024)
    args=parser.parse_args()
    if args.mode=="h50z":
        report=export_h50z_tiles(args.image_dir,args.labels,args.out_dir,
                                tile_size=args.tile_size,include_fiducials=not args.no_fiducials)
        summary={"available_images":report["available_source_images"],
                 "annotated_images_total":report["images_mentioned_in_json"],
                 "crop_count":len(report["tiles"]),
                 "annotation_kind":report["supervision"]}
    else:
        p=args.track_file
        frame_id=p.parent.name
        prefix=frame_id+"_"
        if not p.name.startswith(prefix) or not p.name.endswith(".txt"):
            parser.error("track-file expected <frame>/<frame>_<track_id>.txt")
        track_id=p.name[len(prefix):-4]
        t=parse_track(p,frame_id,track_id)
        def index(s: str) -> int:
            if s=="middle":return t["n_points"]//2
            if s=="last":return t["n_points"]-1
            return int(s)
        indices=tuple(dict.fromkeys(index(s.strip()) for s in args.indices.split(",")))
        report=export_visualmetro_triplet_tiles(t,{1:args.view1,2:args.view2,3:args.view3},
                                                args.out_dir,indices=indices,tile_size=args.tile_size)
        summary={"frame":frame_id,"track_id":track_id,"correspondences":len(indices),
                 "crop_count":3*len(indices),"stereo_calibrated":False}
    zip_path=args.out_dir.with_suffix(".zip")
    _archive(args.out_dir,zip_path)
    summary["zip_path"]=str(zip_path)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
