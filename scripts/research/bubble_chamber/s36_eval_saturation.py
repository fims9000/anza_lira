#!/usr/bin/env python3
"""S36 new reserved cross-photo synthetic controls, including highlight clipping.

Real photo pixels are genuine, but all test labels are artificially injected
measured-shape tracks and artificial scratches, NOT real physical tracks.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s36_saturation_normalization import (
 saturation_blend_input,holdout_exposure,S36_TEST_FORWARD_SEEDS,S36_TEST_REVERSE_SEEDS)
from scripts.research.bubble_chamber.s35_eval_consistency import _model_load,_batched_prob
from scripts.research.bubble_chamber.s32_heldout_injection_control import independent_injections
from scripts.research.bubble_chamber.s33_reverse_validation import generate_holdout_005
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source


def _load_saturation(folder,photo,seed):
    p=Path(folder)/f'azunet_saturation_blend_train{photo}_seed{seed}.pt'
    state=torch.load(p,map_location='cpu',weights_only=True)
    expected={'schema':'s36_machine_only_azunet_saturation_blend_v1',
      'architecture':'azunet','mode':'saturation_blend','source_photo':photo,
      'test_photo_unused':'006' if photo=='005' else '005',
      'train_seed':seed,'human_physical_track_truth':False,'physical_auto_join_authorized':False}
    if any(state.get(k)!=v for k,v in expected.items()):raise ValueError('S36 checkpoint could contain other/photo labels')
    model=build_model('azunet',widths=(4,6,8,12),rules=2)
    model.load_state_dict(state['model_state']);model.eval();return model


def evaluate(tiles,measurements,baseline_dir,new_dir,photo):
    if photo not in ('005','006'):raise ValueError('Invalid photo split')
    verify_model_source();torch.set_num_threads(2)
    seeds=S36_TEST_FORWARD_SEEDS if photo=='005' else S36_TEST_REVERSE_SEEDS
    result={'schema':'s36_saturation_aware_photo_transfer_synthetic_overlay_only_v1',
       'source_train_photo':photo,'source_test_photo':'006' if photo=='005' else '005',
       'test_seeds':list(seeds),'fixed_test_scenarios':['identity','bright','dark','highlight_clip'],
       'source_original_azconv_unchanged':True,'test_photo_not_used_for_training':True,
       'trained_on_human_certified_tracks':False,'natural_h50z_true_tracks_known':False,
       'same_physical_run':True,'auto_accept_physical_join':False,
       'results':[]}
    for j,seed in enumerate(seeds):
        train_seed=(20261025,20261026)[j%2]
        if photo=='005':cases,source=independent_injections(tiles,measurements,seed=seed,n=48)
        else:cases,source=generate_holdout_005(tiles,measurements,seed,48)
        baseline={
            'AZUNet_quantile_S35_matched':(_model_load(baseline_dir,'azunet','matched_two_view',photo,train_seed),'quantile'),
            'U_Net_local_S35_matched':(_model_load(baseline_dir,'unet','matched_two_view',photo,train_seed),'local'),
            'AZUNet_saturation_blend_S36':(_load_saturation(new_dir,photo,train_seed),'saturation_blend')}
        for perturb in result['fixed_test_scenarios']:
            imgs=[holdout_exposure(case.gray,perturb) for case in cases]
            valid=[(case.synthetic_track_mask>0)|(case.synthetic_scratch_mask>0) for case in cases]
            labels=np.concatenate([case.synthetic_track_mask[v].astype(int) for case,v in zip(cases,valid)])
            if set(np.unique(labels))!={0,1}:raise AssertionError('Expected synthetic track vs scratch')
            metrics={};mean_alpha=float(np.mean([saturation_blend_input(x,return_diagnostics=True)[1]['local_branch_weight_uncalibrated'] for x in imgs]))
            clipped=float(np.mean([np.mean((x<=1)|(x>=254)) for x in imgs]))
            for name,(model,mode) in baseline.items():
                if mode=='saturation_blend':
                    scores=_batched_prob_s36(model,imgs)
                else:
                    scores=_batched_prob(model,imgs,mode)
                p=np.concatenate([s[v] for s,v in zip(scores,valid)])
                if len(p)!=len(labels) or not np.isfinite(p).all():raise ValueError('Invalid eval scores')
                metrics[name]={'synthetic_injected_only_AUROC':float(roc_auc_score(labels,p)),
                               'synthetic_injected_only_AUPRC':float(average_precision_score(labels,p))}
            metrics['AZUNet_saturation_blend_S36']['paired_delta_AUROC_vs_quantile']=float(
                metrics['AZUNet_saturation_blend_S36']['synthetic_injected_only_AUROC']-
                metrics['AZUNet_quantile_S35_matched']['synthetic_injected_only_AUROC'])
            result['results'].append({'seed':seed,'train_seed':train_seed,'perturbation':perturb,
             'artificial_scenes':48,'generated_positive_pixel_count':int(labels.sum()),
             'generated_scratch_negative_pixel_count':int(len(labels)-labels.sum()),
             'mean_input_clipped_fraction':clipped,'mean_saturation_blend_local_weight':mean_alpha,
             'metrics':metrics,'data_source_provenance':source if perturb=='identity' else None})
            print('S36 EVAL',photo,seed,perturb,
                {k:round(v['synthetic_injected_only_AUROC'],4) for k,v in metrics.items()},
                'clip',round(clipped,3),'alpha',round(mean_alpha,3),flush=True)
    return result


def _batched_prob_s36(model,images,batch=16):
    result=[]
    with torch.no_grad():
        for i in range(0,len(images),batch):
            inp=np.stack([saturation_blend_input(x) for x in images[i:i+batch]])[:,None]
            scores=torch.sigmoid(model(torch.from_numpy(inp)))[...,0,:,:].numpy()
            result.extend(scores)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for opt in ('tiles','measurements','baselines','saturation-models','out'):
        p.add_argument('--'+opt,required=True,type=Path)
    p.add_argument('--train-photo',choices=('005','006'),required=True)
    a=p.parse_args();report=evaluate(a.tiles,a.measurements,a.baselines,a.saturation_models,a.train_photo)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
