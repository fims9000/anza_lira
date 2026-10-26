#!/usr/bin/env python3
"""S84 private read-only H50Z center crops; no physical GT and no source edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
from scripts.research.bubble_chamber import s84_calibrate_then_fresh_test_graph_context as s84
from scripts.research.bubble_chamber import s80_native_unlabelled_audit as s80n

def run(data:Path,tiles:Path,calibration:Path)->dict:
    if not calibration.is_file():raise FileNotFoundError('Saved frozen S84 CAL needed')
    frozen=json.loads(calibration.read_text())
    if (frozen.get('phase')!='cal' or frozen.get('smoke_NOT_CONFIRMATORY') or
        frozen.get('operator_sha256')!=s84.digest(s84.OPERATOR) or
        frozen.get('protocol_sha256')!=s84.digest(s84.PROTOCOL)):
        raise ValueError('Frozen source-only S84 CAL mismatch')
    templates,split=s84.s72._source_data(data)
    if split!=s84.s80.SPLIT_SHA:raise ValueError('Changed independent source frames')
    source_cal=s84.s75.frozen_s72(s84.s80.S72_CAL)
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if [len(photos[p]) for p in ('005','006')]!=[21,18]:
        raise ValueError('Expected 39 original genuine photo crops')
    counts={arm:dict(candidate_lines=0,proposed_joins_UNKNOWN=0,
            additional_lines=0,extra_considered=0,extra_rejected=0,
            synthetic_GT=0,native_expert_GT=0) for arm in s84.ARMS}
    selected=dict(candidate_lines=0,proposed_joins_UNKNOWN=0,additional_lines=0)
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        direction=f'{source}_to_{photo}'
        scorer,cutoff=s84.s68._fit_source(tiles,source,templates)
        cal=frozen['directions'][direction]
        if float(cal['S56_frozen_CAL_cutoff'])!=float(cutoff):
            raise ValueError('Original graph trained on source drifted')
        old_arm=source_cal['directions'][direction]['CAL_selected']
        chosen=cal['CAL_choice']['CAL_only_selected']
        if chosen not in s84.ARMS:raise ValueError('Unknown CAL selection')
        for path in photos[photo]:
            im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Unreadable private H50Z crop')
            gray=s80n.centre_crop(im)
            preds=s84.infer_from_pixels(gray,scorer,cutoff,old_arm,cal['CAL_only_thresholds'])
            for arm,pred in preds.items():
                c=counts[arm]
                c['candidate_lines']+=len(pred['polylines'])
                c['proposed_joins_UNKNOWN']+=len(pred['chosen'])
                c['additional_lines']+=int(pred['added_pixel_instances'])
                c['extra_considered']+=int(pred['extra_considered'])
                c['extra_rejected']+=int(pred['extra_rejected'])
            p=preds[chosen]
            selected['candidate_lines']+=len(p['polylines'])
            selected['proposed_joins_UNKNOWN']+=len(p['chosen'])
            selected['additional_lines']+=int(p['added_pixel_instances'])
    return {'schema':'S84_39_H50Z_REAL_PHOTOS_MACHINE_HYPOTHESES_ONLY_v1',
            'experiment_id':s84.EXPERIMENT_ID,'tiles':39,'photo_runs':1,
            'visualmetro_film070_is_not_H50Z_GT':True,
            'no_synthetic_insertions':True,'native_expert_GT_count':0,
            'all_native_joins':'UNKNOWN','physical_auto_joins_authorized':False,
            'frozen_CAL_sha256':s84.digest(calibration),'arms':counts,
            'direction_selected_counts':selected}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('No overwriting genuine photo audit')
    result=run(a.visualmetro,a.photo_tiles,a.calibration)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print('S84 genuine native UNKNOWN:',result['arms'],flush=True)
if __name__=='__main__':main()
