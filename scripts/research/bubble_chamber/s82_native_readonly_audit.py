#!/usr/bin/env python3
"""S82 read-only genuine H50Z machine-proposal counts; NO native physics GT."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
from scripts.research.bubble_chamber import s82_calibrate_per_line_then_test as s82
from scripts.research.bubble_chamber import s80_native_unlabelled_audit as s80n

def run(visualmetro:Path,tiles:Path,calibration:Path):
    if not calibration.is_file():raise FileNotFoundError('Need completed immutable S82 CAL JSON')
    cal=json.loads(calibration.read_text())
    if (cal.get('phase')!='cal' or cal.get('smoke_NOT_CONFIRMATORY') or
        cal.get('operator_sha256')!=s82.digest(s82.OPERATOR) or
        cal.get('protocol_sha256')!=s82.digest(s82.PROTOCOL)):
        raise ValueError('CAL version/protocol/inference changed')
    s72f=s82.s75.frozen_s72(s82.s80.S72_CAL)
    templates,split=s82.s72._source_data(visualmetro)
    if split!=s82.s80.SPLIT_SHA:raise ValueError('Source split changed')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:raise ValueError('Need authentic 39 H50Z crops')
    counts={arm:dict(candidate_lines=0,graph_joins_advisory_UNKNOWN=0,
                     additional_lines=0,extra_considered=0,extra_rejected=0,
                     native_verified_GT=0) for arm in s82.ARMS}
    for photo in ('005','006'):
        source='006' if photo=='005' else '005';direction=f'{source}_to_{photo}'
        scorer,cutoff=s82.s68._fit_source(tiles,source,templates)
        frozen=cal['directions'][direction]
        if float(cutoff)!=float(frozen['frozen_S56_cutoff']):
            raise ValueError('Source-fitted S56 cutoff drift')
        graph_arm=s72f['directions'][direction]['CAL_selected']
        thresholds=frozen['CAL_only_local_thresholds']
        for tile in photos[photo]:
            im=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Cannot read genuine private H50Z crop')
            pixels=s80n.centre_crop(im)
            predictions=s82.infer_from_pixels(pixels,scorer,cutoff,graph_arm,thresholds)
            for arm,pred in predictions.items():
                c=counts[arm];c['candidate_lines']+=len(pred['polylines'])
                c['graph_joins_advisory_UNKNOWN']+=len(pred['chosen'])
                c['additional_lines']+=int(pred['added_pixel_instances'])
                c['extra_considered']+=int(pred['extra_considered'])
                c['extra_rejected']+=int(pred['extra_rejected'])
    return {'schema':'S82_H50Z_39_NATIVE_MACHINE_ONLY_UNKNOWN_V1',
            'experiment_id':s82.EXPERIMENT_ID,'native_physics_GT_count':0,
            'all_machine_joins':'UNKNOWN','source_photo_runs':1,'tiles':39,
            'synthetic_injections_used':False,'center_crop_192_of_512':True,
            'physical_auto_joins_authorized':False,
            'frozen_CAL_sha256':s82.digest(calibration),'arms':counts}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('No overwriting private-native machine audit')
    result=run(args.visualmetro,args.photo_tiles,args.calibration)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('S82 native machine-only UNKNOWN counts:',result['arms'])
if __name__=='__main__':main()
