#!/usr/bin/env python3
"""S85 read-only diagnostic on 39 genuine private H50Z center crops.

NO injected synthetic trajectories and NO native track reference/identity.
All 2D candidate joins retain status UNKNOWN. The chosen CAL arm is evaluated
for diagnostic counts only, not promoted into physical auto-tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85
from bubble_chamber.s85_fragment_ranker import ARMS, model_sha256


def audit(visualmetro:Path, photo_tiles:Path, models_path:Path, cal_path:Path):
    if not models_path.is_file() or not cal_path.is_file():
        raise FileNotFoundError('S85 genuine-photo audit requires frozen TRAIN and CAL first')
    trained=json.loads(models_path.read_text(encoding='utf8'))
    calibration=json.loads(cal_path.read_text(encoding='utf8'))
    deps,templates,photos,frozen,ctx=s85._validate_private(visualmetro,photo_tiles)
    s85._model_receipt(trained,ctx)
    for key in ('schema','locks','source_split_sha256','s72_source_code_locks'):
        if calibration.get(key)!=trained.get(key):
            raise ValueError('S85 source CAL and frozen TRAIN models inconsistent: '+key)
    if calibration.get('frozen_TRAIN_models_sha256')!=s85.digest(models_path) or calibration.get('phase')!='cal':
        raise ValueError('S85 source CAL not frozen against private TRAIN model')
    if calibration.get('smoke_NOT_CONFIRMATORY'):
        raise ValueError('Do not use smoke CAL for native H50Z audit')
    scorers=deps[4]
    metrics={arm:{'line_instances':0,'graph_advisory_joins_UNKNOWN':0,
                  'extra_instances':0,'extra_considered':0,'private_native_particle_GT':0}
             for arm in ARMS}
    selected={'line_instances':0,'graph_advisory_joins_UNKNOWN':0,'extra_instances':0}
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        direction=f'{source}_to_{photo}'
        scorer,cutoff=scorers._fit_source(photo_tiles,source,templates)
        arm=frozen['directions'][direction]['CAL_selected']
        if float(cutoff)!=float(calibration['directions'][direction]['frozen_source_cutoff']):
            raise ValueError('S56 cutoff changed after TRAIN/CAL')
        model=trained['models'][direction]['model']
        if model_sha256(model)!=calibration['directions'][direction]['train_model_sha256']:
            raise ValueError('Private trained model coefficients drifted')
        chosen=calibration['directions'][direction]['CAL_decision']['CAL_selected']
        if chosen not in ARMS:raise ValueError('Unknown frozen CAL choice')
        for tile in photos[photo]:
            img=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape!=(512,512):
                raise ValueError('Authentic original 512px H50Z tile required')
            pixels=img[160:352,160:352].copy()
            predictions=s85.predict_blind(pixels,scorer,cutoff,arm,model)
            for key,pred in predictions.items():
                row=metrics[key]
                row['line_instances']+=len(pred['polylines'])
                row['graph_advisory_joins_UNKNOWN']+=len(pred['chosen'])
                row['extra_instances']+=int(pred['added_pixel_instances'])
                row['extra_considered']+=int(pred['extra_considered'])
            p=predictions[chosen]
            selected['line_instances']+=len(p['polylines'])
            selected['graph_advisory_joins_UNKNOWN']+=len(p['chosen'])
            selected['extra_instances']+=int(p['added_pixel_instances'])
    return {'schema':'S85_REAL_39_H50Z_NO_NATIVE_PHYSICS_GT_V1',
            'id':s85.EXPERIMENT_ID,'genuine_private_H50Z_crops':39,
            'physical_runs':1,'independent_physics_GT_count':0,
            'visualmetro_film070_is_not_native_H50Z_GT':True,
            'all_machine_joins':'UNKNOWN','physical_auto_joins_authorized':False,
            'frozen_TRAIN_sha256':s85.digest(models_path),
            'frozen_CAL_sha256':s85.digest(cal_path),
            'aggregates':metrics,'direction_CAL_selected_counts':selected}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--train-models',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable native UNKNOWN audit already exists')
    result=audit(args.visualmetro,args.photo_tiles,args.train_models,args.calibration)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S85 NATIVE UNKNOWN ONLY:',result['aggregates'])

if __name__=='__main__':main()
