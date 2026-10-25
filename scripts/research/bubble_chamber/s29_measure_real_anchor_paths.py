#!/usr/bin/env python3
"""Measure ONLY model-proposed 2D H50Z lines with source integrity checks."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from bubble_chamber.s29_candidate_measurement import measure_candidate


def run(input_json:Path,crops:Path,out:Path):
    obj=json.loads(Path(input_json).read_text())
    if obj.get('schema')!='s28_genuine_h50z_operator_anchor_to_unverified_2D_path_v1':
        raise ValueError('S28 operator-anchored hypotheses required')
    if obj.get('photo_native_track_segment_masks')!=0 or obj.get('linked_physical_particle_identities')!=0:
        raise ValueError('Unsafe provenance: native model proposals are not labels')
    rows=[]
    for record in obj['records']:
        cid=record['crop_id'];path=Path(crops)/(cid+'.png')
        if not path.is_file():raise FileNotFoundError(path)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=record['source_crop_sha256']:
            raise ValueError(f'Authentic BMP crop changed: {cid}')
        photo=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if photo is None or photo.ndim!=2 or photo.dtype!='uint8':
            raise ValueError(f'Not 8-bit original crop: {cid}')
        for method,key in [('classical','classical_candidate_paths'),('original_AZ_assisted','original_AZUNet_assisted_candidate_paths')]:
            candidates=record.get(key)
            if not isinstance(candidates,list):
                raise ValueError('Unexpected S28 path keys '+str(list(record)))
            for i,c in enumerate(candidates):
                result=measure_candidate(photo,c)
                result.update({'crop_id':cid,'photo_id':record['photo_id'],
                               'source_photo_crop_sha256':actual,
                               'method':method,'hypothesis_rank':i+1,
                               'original_operator_start_xy_in_crop':record['genuine_OPERATOR_start_xy_in_crop'],
                               'operator_start_certifies_tracked_line':False})
                rows.append(result)
    response={'schema':'s29_source_locked_unverified_h50z_path_measures_v1',
              'source_S28_sha256':hashlib.sha256(Path(input_json).read_bytes()).hexdigest(),
              'source_photos':['005','006'],
              'physical_runs':1,'expert_confirmed_paths':0,
              'pixel_contour_masks':0,'candidate_measurements':len(rows),
              'rows':rows}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
    return response


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--S28',type=Path,required=True)
    p.add_argument('--crops',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=run(args.S28,args.crops,args.out)
    print('S29 candidate measurements:',r['candidate_measurements'],'human approved:',r['expert_confirmed_paths'])

if __name__=='__main__':main()
