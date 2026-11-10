#!/usr/bin/env python3
"""S106 read-only cross audit for S105: report only by default.

A detected calibration cross is NOT necessarily a false-positive S105 track.
The optional quarantine changes review previews only, never source/GT data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bubble_chamber.s106_fiducial_quarantine import (
    fiducial_quarantine, suppress_visual_only,
)


def run_audit(*,s105_dir:Path, labels:Path, out_dir:Path,
              apply_quarantine:bool=False):
    if out_dir.exists():
        raise FileExistsError(out_dir)
    report=json.loads((s105_dir/'report.json').read_text())
    raw=json.loads(labels.read_text(encoding='utf-8-sig'))
    photo=report['source_name']
    if photo not in raw:
        raise ValueError('No matching source-photo fiducials')
    shape=raw[photo]
    rect=tuple(report['crop_xyxy_source_px'])
    if (int(shape['original_width'])<rect[2]
            or int(shape['original_height'])<rect[3]):
        raise ValueError('Unmatched source-photo dimensions')

    gray=cv2.imread(str(s105_dir/'01_grayscale_visual_only.png'), cv2.IMREAD_GRAYSCALE)
    visual=cv2.imread(str(s105_dir/'13_S105_LONG_RANGE_CONTEXT_SAME_DISPLAY_ID.png'),cv2.IMREAD_COLOR)
    s88=cv2.imread(str(s105_dir/'02_S88_observed_mask.png'),cv2.IMREAD_GRAYSCALE)
    display=cv2.imread(str(s105_dir/'14_S105_CONTEXT_SOURCE_ONLY_MASK.png'),cv2.IMREAD_GRAYSCALE)
    if any(x is None for x in (gray, visual, s88, display)):
        raise ValueError('Missing native-resolution S105 source or display masks')
    if (gray.shape!=tuple(visual.shape[:2]) or gray.shape!=s88.shape
            or gray.shape!=display.shape):
        raise ValueError('Mismatched source/display dimensions')
    if (rect[2]-rect[0],rect[3]-rect[1])!=(gray.shape[1],gray.shape[0]):
        raise ValueError('Mismatched crop coordinate frame')

    marks=[{'label':m['label'],'xy_original_px':[m['x_px'],m['y_px']]}
           for m in shape['keypoints'] if m['label'].startswith('f')
           and rect[0]<=m['x_px']<rect[2] and rect[1]<=m['y_px']<rect[3]]
    proposed,details=fiducial_quarantine(gray,marks,rect)
    visible=display>0
    if apply_quarantine:
        preview,shown=suppress_visual_only(gray,visual,visible,proposed)
    else:
        preview=visual.copy()
        shown=visible.copy()

    out_dir.mkdir(parents=True,exist_ok=False)
    def save(name,img):
        if not cv2.imwrite(str(out_dir/name),img):
            raise OSError(f'Could not write {name}')
    save('00_S105_UNMODIFIED_CONTEXT.png',visual)
    save('01_S106_POSSIBLE_FIDUCIAL_CORRIDOR.png',proposed.astype(np.uint8)*255)
    save('02_S106_REVIEW_PREVIEW.png',preview)
    save('03_S106_DISPLAY_MASK.png',shown.astype(np.uint8)*255)
    save('04_S106_S88_OVERLAP_DIAGNOSTIC_NOT_FALSE_POSITIVES.png',
         ((s88>0)&proposed).astype(np.uint8)*255)
    save('05_S106_S105_DISPLAY_OVERLAP_DIAGNOSTIC.png',
         (visible&proposed).astype(np.uint8)*255)
    with labels.open('rb') as f:
        labels_sha=hashlib.file_digest(f,'sha256').hexdigest()
    summary={
        'schema':'S106_FIDUCIAL_REVIEW_ONLY_BY_DEFAULT_V2',
        'source_name':photo,
        'source_sha256':report['source_sha256'],
        'source_bmp_coordinates':rect,
        'labels_sha256':labels_sha,
        'fiducials_in_crop':len(marks),
        'two_axes_source_supported_count':sum(x['status']=='TWO_AXES_SOURCE_SUPPORTED'
                                              for x in details),
        's88_candidate_px_total':int((s88>0).sum()),
        's88_candidate_px_inside_proposed_corridor':int(((s88>0)&proposed).sum()),
        's105_display_px_total':int(visible.sum()),
        's105_display_px_inside_proposed_corridor':int((visible&proposed).sum()),
        'proposed_quarantine_area_px':int(proposed.sum()),
        's106_quarantine_actually_applied_to_preview':bool(apply_quarantine),
        'source_pixel_or_track_gt_modified':False,
        'overlap_is_not_proven_false_positive':True,
        'potentially_affected_true_tracks_in_cross_corridor':'UNKNOWN',
        'fiducial_details':details,
    }
    assert np.array_equal(shown,visible&~proposed) if apply_quarantine else np.array_equal(shown,visible)
    (out_dir/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--s105-dir',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--apply-quarantine',action='store_true',
                   help='Experimental preview-only suppression; NOT validated false positives')
    a=p.parse_args(argv)
    r=run_audit(s105_dir=a.s105_dir,labels=a.labels,out_dir=a.out_dir,
                apply_quarantine=a.apply_quarantine)
    print(json.dumps({k:v for k,v in r.items() if k!='fiducial_details'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
