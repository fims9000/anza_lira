#!/usr/bin/env python3
"""S54 post-hoc OOD and patch-size provenance audit (machine-only, private)."""
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
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s53_bead_texture import merge_features
from bubble_chamber.s54_context_ood import (SCHEMA,calibration_ood_threshold,
                                             score_context_aligned)
from scripts.research.bubble_chamber.s43_machine_artifact_trial import frozen_frame_folds
from scripts.research.bubble_chamber.s53_benchmark_bead_texture import PROTOCOL,prepare_examples


def run(visualmetro:Path,tiles:Path,s39:Path,s53_private:Path,
        model_dir:Path,out_private:Path,out_public:Path):
    source=json.loads(s39.read_text())
    prior=json.loads(s53_private.read_text())
    if (len(source['tiles'])!=39 or len(prior['tiles'])!=39
        or prior['total_original_candidates_preserved']!=567
        or prior['known_physical_track_truth_count']!=0
        or prior['native_physical_auto_joins']!=0):
        raise ValueError('Unsafe/mismatched H50Z S39/S53 input')
    manifest=audited_large_manifest(visualmetro)
    folds=frozen_frame_folds(manifest)
    config={}
    thresholds={}
    for train_photo,test_photo,seed in [('005','006',20265303),('006','005',20265313)]:
        mdir=model_dir/f'{train_photo}_to_{test_photo}_{seed}'
        models={name:json.loads((mdir/(name+'_simulator_only.json')).read_text())
                for name in ('existing_S43','S43_plus_bead_texture_S53')}
        calibration=prepare_examples(manifest,folds['calibration'],tiles,
                 photo=train_photo,n=PROTOCOL['calibration_scenes_per_direction'],
                 seed=PROTOCOL['calibration_seeds'][train_photo],test=False)
        thresh={name:calibration_ood_threshold(
            merge_features(calibration['X43'],calibration['X53']) if 'S53' in name
            else calibration['X43'],m,quantile=.99)
                for name,m in models.items()}
        config[test_photo]=models
        thresholds[test_photo]=thresh
    outputs=[];roll=defaultdict(lambda:Counter());deltas=defaultdict(list)
    values=defaultdict(list);dominant=defaultdict(Counter);ids=set()
    for tile,old in zip(source['tiles'],prior['tiles']):
        cid=tile['crop_id']; photo=tile['photo_id']
        if (cid,photo)!=(old['crop_id'],old['photo']) or cid in ids:
            raise ValueError('Wrong H50Z tile mapping or duplicate')
        ids.add(cid)
        gray=cv2.imread(str(tiles/(cid+'.png')),cv2.IMREAD_GRAYSCALE)
        if gray is None or sha256(gray.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Private H50Z pixel corruption / bad provenance')
        records=score_context_aligned(gray,tile['proposals'],photo=photo,
             checkpoints=config[photo],thresholds=thresholds[photo])
        old_by_id={r['draft_id']:r for r in old['candidates']}
        if len(old_by_id)!=len(old['candidates']) or len(records)!=len(old_by_id):
            raise ValueError('Native proposal was lost, duplicated or invented')
        for r in records:
            p=old_by_id[r['draft_id']]
            if p.get('candidate_preserved') is not True or p.get('physical_auto_join_authorized') is not False:
                raise ValueError('Prior machine status unsafe')
            roll[photo][r['status']]+=1
            if r['status']=='UNKNOWN_UNSCORABLE_CONTEXT':continue
            dominant[photo][r['S53_dominant_outlier_feature']]+=1
            delta=abs(r['S53_simulated_origin_score_192px_NOT_TRACK_PROB']-
                    p['S53_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB'])
            r['absolute_change_from_512px_score_SOURCE_CONTEXT_ONLY']=round(delta,6)
            deltas[photo].append(delta)
            values[photo].append(r['S53_simulated_origin_score_192px_NOT_TRACK_PROB'])
        outputs.append({'crop_id':cid,'photo':photo,
                        'source_gray_SHA256':tile['source_gray_sha256'],
                        'candidates':records})
    if sum(sum(q.values()) for q in roll.values())!=567:
        raise AssertionError('Native machine candidate preservation failed')
    private={'schema':SCHEMA,'source_S53_full512_private_reference':s53_private.name,
             'native_expert_GT_count':0,'native_physical_auto_joins':0,
             'all_machine_proposals_preserved':567,
             'thresholds_calibrated_only_on_SOURCE_SYNTHETIC_CALIBRATION':thresholds,
             'tiles':outputs}
    out_private.parent.mkdir(parents=True,exist_ok=True)
    out_private.write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n')
    public={'schema':SCHEMA,'context_diagnostic_is_posthoc_NOT_confirmatory':True,
            'native_H50Z_physical_GT_count':0,'physical_auto_joins':0,
            'all_567_original_candidates_preserved':True,
            'fresh_independent_physical_run_count':0,
            'real_photo_run_count':1,
            'source_calibration_synthetic_frames':len(folds['calibration']),
            'source_192_patch_size_and_12px_margin':True,
            'simulator_calibrated_99pct_max_abs_z_is_not_real_OOD_guarantee':True,
            'source_model_training_uses_only_opposite_photo':True,
            'counts':{}}
    for photo in ['005','006']:
        x=roll[photo];dv=np.asarray(deltas[photo],dtype=float);vv=np.asarray(values[photo],dtype=float)
        public['counts'][photo]={
            'all_native_machine_candidates':int(sum(x.values())),
            'cannot_fit_source_192_context_without_resampling':int(x['UNKNOWN_UNSCORABLE_CONTEXT']),
            'scorable_candidates':int(len(dv)),
            'within_source_simulator_calibration_support_NOT_validated_tracks':int(x['UNKNOWN_WITHIN_SIMULATOR_SUPPORT_ONLY']),
            'outside_source_simulator_calibration_support_NOT_artifact_truth':int(x['UNKNOWN_OUTSIDE_CALIBRATION_SUPPORT']),
            'S53_score_median_192_context_SIMULATOR_ONLY':float(np.median(vv)),
            'S53_score_change_from_512_context_median_abs_NOT_accuracy':float(np.median(dv)),
            'S53_score_change_from_512_context_gt_0p2':int(np.count_nonzero(dv>.20)),
            'most_frequent_S53_outlier_features':dominant[photo].most_common(5),
        }
    out_public.parent.mkdir(parents=True,exist_ok=True)
    out_public.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    for key in ('visualmetro','photo_tiles','s39_private','s53_native_private','models','private_out','public_out'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    r=run(a.visualmetro,a.photo_tiles,a.s39_private,a.s53_native_private,
          a.models,a.private_out,a.public_out)
    print(json.dumps(r,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
