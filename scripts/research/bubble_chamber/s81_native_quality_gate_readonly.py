#!/usr/bin/env python3
"""S81 read-only quality-gated machine proposals on original private H50Z tiles.

NO native physical GT exists; all proposed lines and joins stay UNKNOWN.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s80_native_unlabelled_audit as s80n
from scripts.research.bubble_chamber import s81_calibrate_quality_gate as s81


def run(visualmetro:Path,tiles:Path,calibration:Path)->dict:
    if not calibration.is_file():raise FileNotFoundError('Need frozen independent S81 source-CAL')
    frozen=json.loads(calibration.read_text())
    if frozen.get('phase')!='cal' or frozen.get('smoke_NOT_CONFIRMATORY'):
        raise ValueError('Cannot use a smoke or TEST dataset to calibrate genuine images')
    if frozen.get('protocol_sha256')!=s81.digest(s81.PROTOCOL) or frozen.get('operator_sha256')!=s81.digest(s81.OPERATOR):
        raise ValueError('Frozen S81 CAL algorithm/protocol changed')
    s72_reference=s75.frozen_s72(s81.s80.S72_CAL)
    templates,split=s72._source_data(visualmetro)
    if split!=s81.s80.SPLIT_SHA:raise ValueError('Original source split changed')
    files={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if [len(files[x]) for x in ('005','006')]!=[21,18]:raise ValueError('Original 39 genuine photo tiles required')
    counts={arm:{'candidate_lines':0,'graph_joins_advisory_UNKNOWN':0,
                 'additional_lines':0,'quality_gate_triggered':0,'native_GT_count':0}
            for arm in s81.ARMS}
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        direction=f'{source}_to_{photo}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        if float(cutoff)!=float(frozen['directions'][direction]['S56_cutoff']):
            raise ValueError('Frozen source-CAL scorer changed')
        graph_arm=s72_reference['directions'][direction]['CAL_selected']
        thresholds=frozen['directions'][direction]['CAL_only_quality_thresholds']
        for tile in files[photo]:
            im=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Bad private photo pixels')
            pix=s80n.centre_crop(im)
            pred=s81.infer_from_pixels(pix,scorer,cutoff,graph_arm,thresholds)
            for arm,data in pred.items():
                count=counts[arm]
                count['candidate_lines']+=len(data['polylines'])
                count['graph_joins_advisory_UNKNOWN']+=len(data['chosen'])
                count['additional_lines']+=int(data['added_pixel_instances'])
                count['quality_gate_triggered']+=int(data['quality_gate_triggered'])
    return {'schema':'S81_H50Z_39_NATIVE_UNKNOWN_MACHINE_ONLY_V1',
            'experiment_id':s81.EXPERIMENT_ID,'central_192px_of_512px':True,
            'unlabelled_private_tiles':39,'independent_physical_runs':False,
            'native_particle_GT_count':0,'all_graph_proposals':'UNKNOWN',
            'physical_auto_joins_authorized':False,'calibration_sha256':s81.digest(calibration),
            'arms':counts}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Read-only native audit immutable output exists')
    report=run(a.visualmetro,a.photo_tiles,a.calibration)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S81 genuine H50Z machine proposals, NO GT:',json.dumps(report['arms']),flush=True)
if __name__=='__main__':main()
