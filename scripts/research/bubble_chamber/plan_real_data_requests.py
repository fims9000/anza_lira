#!/usr/bin/env python3
"""Prepare small BMP requests and annotate fiducial QC without changing data."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.film_data_planner import audit_fiducials,build_crop_request,save_plan


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--h50z-labels',required=True,type=Path)
    p.add_argument('--visualmetro-root',required=True,type=Path)
    p.add_argument('--available-bmps',type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--per-new-frame',type=int,default=6)
    args=p.parse_args()
    qc=audit_fiducials(args.h50z_labels)
    plan=build_crop_request(args.h50z_labels,args.visualmetro_root,available_root=args.available_bmps,per_new_frame=args.per_new_frame)
    save_plan(plan,qc,args.out_dir)
    print('frames with duplicate IDs:',len(qc['duplicate_id_frames']))
    print('frames with outlier IDs:',len(qc['outlier_id_frames']))
    print('new H50Z images requested:',','.join(x['image'] for x in plan['h50z_image_priority']))
    print('small PNG regions requested:',len(plan['requests']))
    print('output:',args.out_dir)

if __name__=='__main__':main()
