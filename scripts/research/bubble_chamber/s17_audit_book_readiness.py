#!/usr/bin/env python3
"""Audit whether visualMetro/Hadroide BOOK procedures can be applied to real BMPs.

A start landmark is NOT a manually confirmed stopping point. Three BMPs plus
run-specific optics/field and expert correspondence are prerequisites for a
physical 3D fit. This command only inspects metadata; never labels pixels.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

PATTERN=re.compile(r'^(?P<camera>[A-Za-z0-9]+)_(?P<film>\d{3})_(?P<view>\d{2})_(?P<frame>\d{3})\.bmp$',re.IGNORECASE)


def audit(labels_file:Path, bmp_root:Path, *, optical_calibration:Path|None=None,
          fiducial_reference:Path|None=None, magnetic_field:Path|None=None)->dict:
    root=Path(bmp_root).resolve();labels=json.loads(Path(labels_file).read_text())
    if not isinstance(labels,dict):raise ValueError('invalid labels metadata')
    groups=defaultdict(lambda:{'views':set(),'files':[],'start_anchors':0,'fiducial_marks':0})
    for name,meta in sorted(labels.items()):
        match=PATTERN.fullmatch(name)
        if not match or Path(name).name!=name:raise ValueError('unsafe/unrecognized film filename '+name)
        photo=root/name
        if not photo.is_file():continue
        fil=match.group('film');frame=match.group('frame');view=int(match.group('view'))
        key=(match.group('camera'),fil,frame)
        if view not in range(1,7):raise ValueError('unsupported view index')
        g=groups[key]
        if view in g['views']:raise ValueError('duplicate per-frame projection')
        g['views'].add(view)
        g['files'].append({'view':view,'filename':name,'size_bytes':photo.stat().st_size,
                           'sha256':sha256(photo.read_bytes()).hexdigest()})
        g['start_anchors']+=sum(k['label']=='start' for k in meta['keypoints'])
        g['fiducial_marks']+=sum(k['label'].startswith('f') for k in meta['keypoints'])
    cal={}
    for key,path in [('optical_model',optical_calibration),('fiducial_layout',fiducial_reference),('magnetic_field_model',magnetic_field)]:
        p=Path(path).resolve() if path is not None else None
        cal[key]={'path':str(p),'sha256':sha256(p.read_bytes()).hexdigest(),'status':'PRESENT_NOT_VALIDATED'} if p and p.is_file() else {'status':'MISSING'}
    events=[]
    for (camera,fil,frame),info in sorted(groups.items()):
        views=sorted(info['views'])
        stereo=(views==[1,2,3] or views==[4,5,6])
        events.append({'camera':camera,'film':fil,'frame':frame,'views_present':views,
            'three_views_same_frame':stereo,'start_keypoints_not_stops':info['start_anchors'],
            'fiducial_keypoints':info['fiducial_marks'],'source_bmps':info['files'],
            'can_assign_expert_pixel_masks':False,
            'can_run_book_physical_mass_fit':False,
            'blocking_conditions':(["OTHER_TWO_PROJECTIONS_NOT_AVAILABLE"] if not stereo else [])+
              (["OPTICAL_FIDUCIAL_OR_BFIELD_CALIBRATION_UNVERIFIED"] if not all(c['status'].startswith('PRESENT') for c in cal.values()) else [])+
              ['STOPPING_PARTICLE_AND_STEREO_TRACK_CORRESPONDENCE_NOT_EXPERT_CONFIRMED']})
    result={'schema':'s17_book_real_readiness_v1','reference':{
        'visualmetro':'Belyaev Bushmina Klevtsova 2025 guide, 3 BMPs + fiducials + run-specific calibration',
        'hadroide':'Belyaev P1-2023-22, limited near confirmed stopping tracks; cannot derive real masks from equations'},
        'available_bmps':sum(len(x['source_bmps']) for x in events),
        'events_with_any_bmp':len(events), 'complete_three_view_events':sum(x['three_views_same_frame'] for x in events),
        'available_starts':sum(x['start_keypoints_not_stops'] for x in events),
        'available_fiducials':sum(x['fiducial_keypoints'] for x in events),
        'calibration':cal,'events':events,
        'expert_track_mask_count':0,'track_identity_count':0,
        'warning':'START is a named keypoint, NOT proof of a stopping point; fiducials are not particle tracks.',
        'conclusion':'PHYSICS_CURVES_CAN_PRETRAIN_SYNTHETIC_ONLY; REAL_PHOTO_TRACK_GT_UNAVAILABLE'}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--bmp-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    opts=p.parse_args()
    x=audit(opts.labels,opts.bmp_dir)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    opts.out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in x.items() if k not in {'events','calibration','reference'}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
