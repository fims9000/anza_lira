#!/usr/bin/env python3
"""Inventory of large BMPs, H50Z crosses and separate visualMetro sparse tracks.

Exports aggregate-only public counts and optional strictly PRIVATE source
index/previews. It NEVER turns crosses into physical track masks or matches
film070 images to H50Z automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.physicists_intake import inventory_sources, preview_crosses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bmp-root',type=Path,help='Folder of 8-bit large BMP files, recursive, header-only')
    parser.add_argument('--crosses',type=Path,help='H50Z labels.json with f1..f8/start marks; NOT masks')
    parser.add_argument('--visualmetro-root',type=Path,help='Extracted 070_### measured tracks (not H50Z GT)')
    parser.add_argument('--stereo-map',type=Path,help='Operator-verified 3-view map; never auto-pair films')
    parser.add_argument('--public-summary',type=Path,required=True,help='Safe aggregate count JSON only')
    parser.add_argument('--private-index',type=Path,help='PRIVATE file names, mark coordinates, mappings')
    parser.add_argument('--private-cross-preview',type=Path,help='PRIVATE operator crosses on pixel crops')
    parser.add_argument('--hash-bmps',action='store_true',help='Optional SHA-256 all full images; slow')
    parser.add_argument('--tile-size',type=int,default=512)
    parser.add_argument('--preview-limit',type=int,default=30)
    args=parser.parse_args(argv)
    if args.stereo_map and not (args.bmp_root and args.visualmetro_root and args.private_index):
        parser.error('--stereo-map needs --bmp-root, --visualmetro-root and --private-index')
    if args.private_cross_preview and not (args.bmp_root and args.crosses):
        parser.error('--private-cross-preview requires --bmp-root and --crosses')
    root=ROOT.resolve()
    outputs=[x for x in (args.public_summary,args.private_index,args.private_cross_preview) if x is not None]
    if len({str(x.resolve()) for x in outputs})!=len(outputs):
        parser.error('Output destinations must differ')
    for target in (args.private_index,args.private_cross_preview):
        if target is not None and (target.resolve()==root or root in target.resolve().parents):
            parser.error('Private outputs must stay outside the public Git repository')
    if args.public_summary.exists():
        raise FileExistsError('Refuse overwriting public aggregate result')
    if args.private_index and args.private_index.exists():
        raise FileExistsError('Refuse overwriting private source index')
    if args.private_cross_preview and args.private_cross_preview.exists() and any(args.private_cross_preview.iterdir()):
        raise FileExistsError('Refuse overwriting private source preview directory')
    report,secret=inventory_sources(bmp_root=args.bmp_root,crosses=args.crosses,
                                    visualmetro=args.visualmetro_root,stereo_map=args.stereo_map,
                                    hash_bmps=args.hash_bmps)
    if args.private_cross_preview:
        preview_crosses(bmp_root=args.bmp_root,private=secret['H50Z'],
                        output=args.private_cross_preview,
                        tile_size=args.tile_size,limit=args.preview_limit)
    if args.private_index:
        args.private_index.parent.mkdir(parents=True,exist_ok=True)
        args.private_index.write_text(json.dumps(secret,ensure_ascii=False,indent=2)+'\n')
        args.private_index.chmod(0o600)
    args.public_summary.parent.mkdir(parents=True,exist_ok=True)
    args.public_summary.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
