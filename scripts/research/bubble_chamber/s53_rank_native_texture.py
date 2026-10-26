#!/usr/bin/env python3
"""Private native-only S53 model-score audit; no physical GT or auto-linking."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s53_native_texture_advisory import score_native_tile
from bubble_chamber.s53_bead_texture import SCHEMA


def run(s39_path:Path, tiles:Path, model_dirs:Path, private_out:Path,
        public_out:Path, image_out:Path|None=None)->dict:
    s39=json.loads(s39_path.read_text())
    if (s39.get('native_expert_gt_created') != 0 or
        len(s39.get('tiles',[]))!=39 or
        s39.get('auto_physical_join_allowed') is not False):
        raise ValueError('The native machine-only dataset has lost its safe provenance')
    original=0;rows=[];all_scores=defaultdict(list)
    images=0
    if image_out:image_out.mkdir(parents=True,exist_ok=True)
    models_by_target={}
    for source,target,seed in [('005','006',20265303),('006','005',20265313)]:
        base=model_dirs/f'{source}_to_{target}_{seed}'
        models_by_target[target]={key:json.loads((base/(key+'_simulator_only.json')).read_text())
                                 for key in ('existing_S43','S43_plus_bead_texture_S53')}
    seen_ids=set()
    for t in s39['tiles']:
        cid=t['crop_id'];photo=t['photo_id']
        if cid in seen_ids:raise ValueError('Duplicated native crop identity')
        seen_ids.add(cid)
        img=cv2.imread(str(tiles/(cid+'.png')),cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape!=(512,512) or sha256(img.tobytes()).hexdigest()!=t['source_gray_sha256']:
            raise ValueError('H50Z private image source checksum mismatch')
        preds=score_native_tile(img,t['proposals'],photo=photo,
                                checkpoints=models_by_target[photo])
        if len(preds)!=len(t['proposals']):raise AssertionError('Candidate was silently removed')
        for z in preds:
            all_scores[photo].append((z['S43_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB'],
                                      z['S53_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB']))
        original+=len(preds)
        rows.append({'crop_id':cid,'photo':photo,'native_image_SHA256':t['source_gray_sha256'],
                     'candidates':preds,'number_machine_lines_unchanged':len(preds),
                     'native_particle_GT':0,'auto_physical_join':0})
        if image_out:
            # Native pixels plus overlay are private; not suitable for public Git.
            canvas=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            for p,r in zip(t['proposals'],preds):
                s=r['S53_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB']
                # Colors encode simulator response only, NEVER a physics class.
                col=(40,185,220) if s < .50 else (145,200,55)
                cv2.polylines(canvas,[np.rint(np.asarray(p['points_xy'])).astype('int32')],
                              False,col,2,cv2.LINE_AA)
            cv2.imwrite(str(image_out/(cid+'_S53_SIMULATOR_RESPONSE.jpg')),canvas,
                        [cv2.IMWRITE_JPEG_QUALITY,85])
            images+=1
    if original!=567:raise AssertionError('Canonical 567 native candidates were changed')
    private={'schema':SCHEMA,'native_H50Z_source':'real private BMP crops',
             'known_physical_track_truth_count':0,'native_physical_auto_joins':0,
             'no_native_labels_created':True,'total_original_candidates_preserved':original,
             'tiles':rows}
    private_out.parent.mkdir(parents=True,exist_ok=True)
    private_out.write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n')
    public={'schema':SCHEMA,'native_H50Z_GT_count':0,'independent_physical_runs':1,
      'all_native_candidates_preserved':original,'source_crops_verified_SHA256':len(seen_ids),
      'native_physical_auto_joins':0,'new_native_track_labels':0,
      'scores_are_simulator_origin_similarity_NOT_particle_probability':True,
      'both_photo_synthetic_training_oracle_label_leakage':False,
      'summary':{}}
    for photo,values in all_scores.items():
        a=np.asarray(values)
        public['summary'][photo]={
            'native_machine_candidate_count':len(a),
            'S43_simulator_response_median':float(np.median(a[:,0])),
            'S53_simulator_response_median':float(np.median(a[:,1])),
            'absolute_S53_minus_S43_difference_median':float(np.median(np.abs(a[:,1]-a[:,0]))),
            'raw_score_disagreement_gt_0p2_SIMULATOR_ONLY':int(np.count_nonzero(np.abs(a[:,1]-a[:,0])>.2))}
    public['private_preview_count']=images
    public_out.parent.mkdir(parents=True,exist_ok=True)
    public_out.write_text(json.dumps(public,indent=2,ensure_ascii=False)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--s39-private',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--private-out',type=Path,required=True)
    p.add_argument('--public-out',type=Path,required=True)
    p.add_argument('--private-overlays',type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.s39_private,a.photo_tiles,a.model_dir,a.private_out,
                         a.public_out,a.private_overlays),indent=2),flush=True)
if __name__=='__main__':main()
