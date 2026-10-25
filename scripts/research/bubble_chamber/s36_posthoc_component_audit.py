#!/usr/bin/env python3
"""POST-HOC S36 decomposition; explanatory only, not independent confirmation.

This explicitly separates what changed because the AZUNet was RETRAINED
from what changed because of switching quantile -> saturation-blended input.
The same S36 injections have already been inspected: DO NOT use this script to
choose or calibrate new hyperparameters and report it as held-out evidence.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s36_saturation_normalization import S36_TEST_FORWARD_SEEDS,S36_TEST_REVERSE_SEEDS,holdout_exposure
from scripts.research.bubble_chamber.s35_eval_consistency import _model_load,_batched_prob
from scripts.research.bubble_chamber.s36_eval_saturation import _load_saturation,_batched_prob_s36
from scripts.research.bubble_chamber.s32_heldout_injection_control import independent_injections
from scripts.research.bubble_chamber.s33_reverse_validation import generate_holdout_005


def audit(tiles,measurements,baselines,satmodels,train_photo,n=48):
    torch.set_num_threads(2)
    seeds=S36_TEST_FORWARD_SEEDS if train_photo=='005' else S36_TEST_REVERSE_SEEDS
    out={'schema':'s36_posthoc_model_training_vs_preprocess_decomposition_v1',
      'EXPLORATORY_POST_TEST_NOT_INDEPENDENT_CONFIRMATION':True,
      'real_native_track_truth_available':False,'training_photo':train_photo,
      'source_photos_same_physical_run':True,'physical_auto_join_authorized':False,'results':[]}
    for i,seed in enumerate(seeds):
        train_seed=(20261025,20261026)[i%2]
        if train_photo=='005':cases,_=independent_injections(tiles,measurements,seed=seed,n=n)
        else:cases,_=generate_holdout_005(tiles,measurements,seed,n)
        q=_model_load(baselines,'azunet','matched_two_view',train_photo,train_seed)
        s=_load_saturation(satmodels,train_photo,train_seed)
        for perturb in ('identity','highlight_clip'):
            imgs=[holdout_exposure(r.gray,perturb) for r in cases]
            valid=[(r.synthetic_track_mask>0)|(r.synthetic_scratch_mask>0) for r in cases]
            labels=np.concatenate([r.synthetic_track_mask[m].astype(int) for r,m in zip(cases,valid)])
            methods={
              'quantile_trained_on_quantile':_batched_prob(q,imgs,'quantile'),
              'quantile_trained_on_blend':_batched_prob_s36(q,imgs),
              'blend_trained_on_quantile':_batched_prob(s,imgs,'quantile'),
              'blend_trained_on_blend':_batched_prob_s36(s,imgs)}
            auc={name:float(roc_auc_score(labels,np.concatenate([p[m] for p,m in zip(probs,valid)]))) for name,probs in methods.items()}
            out['results'].append({'seed':seed,'mode':perturb,'AUROC_generated_only':auc,
               'delta_inference_only':auc['quantile_trained_on_blend']-auc['quantile_trained_on_quantile'],
               'delta_training_only':auc['blend_trained_on_quantile']-auc['quantile_trained_on_quantile'],
               'delta_full':auc['blend_trained_on_blend']-auc['quantile_trained_on_quantile']})
        print('S36 POSTHOC',train_photo,seed,'identity Δ',out['results'][-2]['delta_full'],flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('tiles','measurements','baselines','saturation-models','out'):
        p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--train-photo',required=True,choices=('005','006'))
    a=p.parse_args();r=audit(a.tiles,a.measurements,a.baselines,a.saturation_models,a.train_photo)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
