#!/usr/bin/env python3
"""Create small annotated PNG submission from large original BMPs."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.small_bundle import create_small_bundle

p=argparse.ArgumentParser()
p.add_argument('--requests',type=Path,required=True)
p.add_argument('--h50z-bmp-dir',type=Path)
p.add_argument('--visualmetro-bmp-dir',type=Path)
p.add_argument('--visualmetro-view-map',type=Path)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--strict',action='store_true')
a=p.parse_args()
m=create_small_bundle(a.requests,a.out,h50z_root=a.h50z_bmp_dir,
                      visualmetro_root=a.visualmetro_bmp_dir,
                      visualmetro_view_map=a.visualmetro_view_map,strict=a.strict)
print(f"Saved {m['tile_count']} PNGs; {m['missing_count']} requests without matching BMP")
