#!/usr/bin/env python3
"""Source-locked S106 read-only fiducial-cross quarantine after S105 preview."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s106_fiducial_quarantine import fiducial_quarantine,suppress_visual_only


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s105-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    if args.out_dir.exists():raise FileExistsError(args.out_dir)
    report=json.loads((args.s105_dir/'report.json').read_text())
    raw=json.loads(args.labels.read_text(encoding='utf-8-sig'))
    photo=report['source_name']
    if photo not in raw:raise ValueError('no fiducial metadata for the exact BMP')
    shape=raw[photo]
    crop=report['crop_xyxy_source_px']
    if (int(shape['original_width'])<crop[2] or int(shape['original_height'])<crop[3]):
        raise ValueError('unmatched original BMP dimensions')
    gray=cv2.imread(str(args.s105_dir/'01_grayscale_visual_only.png'),cv2.IMREAD_GRAYSCALE)
    original=cv2.imread(str(args.s105_dir/'13_S105_LONG_RANGE_CONTEXT_SAME_DISPLAY_ID.png'),cv2.IMREAD_COLOR)
    src=cv2.imread(str(args.s105_dir/'02_S88_observed_mask.png'),cv2.IMREAD_GRAYSCALE)
    if gray is None or original is None or src is None:raise ValueError('missing S105 source masks')
    marks=[{'label':m['label'],'xy_original_px':[m['x_px'],m['y_px']]}
           for m in shape['keypoints'] if m['label'].startswith('f')
           and crop[0]<=m['x_px']<crop[2] and crop[1]<=m['y_px']<crop[3]]
    q,details=fiducial_quarantine(gray,marks,tuple(crop))
    clean,mask=suppress_visual_only(gray,original,src>0,q)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    def save(name,img):
        if not cv2.imwrite(str(args.out_dir/name),img):raise OSError(name)
    save('00_ORIGINAL_S105_CONTEXT.png',original)
    save('01_S106_SOURCE_FIDUCIAL_QUARANTINE.png',q.astype(np.uint8)*255)
    save('02_S106_TRACK_CANDIDATES_OUTSIDE_FIDUCIAL.png',clean)
    save('03_S106_OBSERVED_ONLY_REVIEW_MASK.png',mask.astype(np.uint8)*255)
    # The quarantined source pixels are saved in a SEPARATE, not-negative layer.
    save('04_S106_REVIEW_QUARANTINED_SOURCE_PIXELS.png',((src>0)&q).astype(np.uint8)*255)
    summary={'schema':'S106_OPERATOR_FIDUCIAL_DISPLAY_QUARANTINE_V1',
             'source_name':photo,'source_sha256':report['source_sha256'],
             'crop_xyxy_source_px':crop,
             'labels_sha256':hashlib.file_digest(args.labels.open('rb'),'sha256').hexdigest(),
             'fiducials_in_crop':len(marks),'verified_cross_geometry_count':sum(x['status']=='TWO_AXES_SOURCE_SUPPORTED' for x in details),
             'S105_source_pixels':int((src>0).sum()),
             'S106_source_pixels_quarantined':int(((src>0)&q).sum()),
             'S106_source_pixels_outside_fiducial':int(mask.sum()),
             'mark_diagnostics':details,
             'source_BMP_and_S105_masks_unchanged':True,
             'quarantine_is_unknown_not_background':True,'confirmed_particle_labels':False}
    assert summary['S106_source_pixels_quarantined']+summary['S106_source_pixels_outside_fiducial']==summary['S105_source_pixels']
    (args.out_dir/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='mark_diagnostics'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
