#!/usr/bin/env python3
"""Freeze S33/S34 aggregates from per-seed controlled synthetic injections.

All datasets are UNLABELED native H50Z; metrics refer exclusively to synthetic
insertions over genuine source pixels and do not imply real particle accuracy.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compile_results(results:Path):
    results=Path(results)
    directions={
      '005_train_006_test':[20261029,20261030,20261031,20261032],
      '006_train_005_test':[20261037,20261038,20261039,20261040],
    }
    out={'schema':'s33_s34_photometric_and_axial_research_2026_10_25_v1',
       'original_AZConv_source_changed':False,
       'native_H50Z_physical_track_truth':False,'source_original_photo_count':2,
       'source_physical_runs':1,'source_H50Z_512px_crops':39,
       'machine_positive_crops':{'005':9,'006':4},
       'real_expert_confirmed_track_masks':0,'real_expert_confirmed_relation_pairs':0,
       'training_machine_marks_only':True,
       'train_photo_test_photo_disjoint':True,
       'injection_seeds_not_independent_physical_runs':True,
       'photometric_preprocessing_fit_free_per_patch':True,
       'photo_006_never_used_for_forward_checkpoint_selection':True,
       'photo_005_never_used_for_reverse_checkpoint_selection':True,
       'model_widths':[4,6,8,12],'fuzzy_rules':2,
       'epochs_fixed':2,'train_seed_count':2,
       'directions':{},'S34_axial':{},
       'source_candidate_evidence':{},
       'interpretation':'Photo normalization reduces gate sensitivity to brightness; real bubble-chamber track recognition unverified',
       'physics_autonomous_join_permitted':False}
    for direction,seeds in directions.items():
        docs=[]
        for seed in seeds:
            f=results/(f'S33_eval_seed{seed}.json' if direction.startswith('005_train') else f'S33_reverse_seed{seed}.json')
            x=json.loads(f.read_text())
            if direction.startswith('005_train'):
                assert x['schema']=='s33_matched_heldout_synthetic_injection_v1'
                results_by_perturb={r['perturbation']:r for r in x['evaluation']}
                key='auroc_injected_only'
            else:
                assert x['schema']=='s33_reverse_006_train_005_test_synthetic_only_v1'
                results_by_perturb={r['perturbation']:r for r in x['evaluations']}
                key='AUROC_synthetic_only'
            docs.append((seed,results_by_perturb,key))
        methods=[f'{arch}_{mode}' for arch in ('azunet','unet') for mode in ('raw','quantile','local')]
        entries={}
        for name in methods:
            vals=[d['identity']['metrics'][name][k] for _,d,k in docs]
            pert={mode:[d[mode]['metrics'][name][k] for _,d,k in docs] for mode in ('bright','dark')}
            baseline=name.split('_')[0]+'_raw'
            raw=[d['identity']['metrics'][baseline][k] for _,d,k in docs]
            entries[name]={'mean_AUROC':round(float(np.mean(vals)),6),
                'AUROC_per_injection_seed':[round(float(v),6) for v in vals],
                'mean_delta_vs_raw':round(float(np.mean(np.asarray(vals)-np.asarray(raw))),6),
                'positive_delta_seed_count':int(sum(a>b for a,b in zip(vals,raw))),
                'mean_abs_brightness_shift_AUROC':round(float(np.mean([abs(a-b) for aug in pert.values() for a,b in zip(aug,vals)])),6),
                'not_native_real_photo_accuracies':True}
        out['directions'][direction]={'seeds':seeds,'per_seed_synthetic_injections':48,
                                      'target_from_artificial_track_vs_scratch_only':True,
                                      'methods':entries}
    axis=[json.loads((results/f'S34_eval_seed{s}.json').read_text()) for s in (20261033,20261034,20261035,20261036)]
    deltas=[x['results'][0]['paired_delta_axis_minus_plain']['AUROC'] for x in axis]
    first=[x['results'][0]['metrics']['azunet_local']['AUROC_injected_only'] for x in axis]
    altered=[x['results'][0]['metrics']['azunet_local_axis']['AUROC_injected_only'] for x in axis]
    out['S34_axial']={'new_unseen_injection_seeds':[20261033,20261034,20261035,20261036],
       'same_train_photo':'005','same_test_photo':'006',
       'lambda_axis_pre_registered':.03,'mean_baseline_AUROC':float(np.mean(first)),
       'mean_axial_AUROC':float(np.mean(altered)),
       'delta_AUROC_per_seed':[float(x) for x in deltas],
       'positive_delta_seed_count':sum(x>0 for x in deltas),
       'mean_delta_AUROC':float(np.mean(deltas)),
       'conclusion':'No consistent axial improvement, do not enable by default'}
    g=json.loads((results/'S33_gate_diagnostic.json').read_text())
    out['AZConv_first_rule_diagnostic']=g
    e=json.loads((results/'S33_133_candid_evidence_PRIVATE.json').read_text())
    assert e['merged_reviewer_drafts']==e['original_draft_count']==133 and not e['physics_auto_join_authorized']
    out['source_candidate_evidence']={'total_unreviewed_model_drafts':e['merged_reviewer_drafts'],
        'original_H50Z_review_crops':e['source_crops'],
        'quantile_model_center_vs_flank':e['summary']['quantile'],
        'local_model_center_vs_flank':e['summary']['local'],
        'no_human_certification':True,
        'candidate_selection_bias':'Original candidates already selected by darkness; positive center-flank score cannot be interpreted as accuracy'}
    return out


def make_charts(results,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9.4,5.0))
    selected=['azunet_raw','azunet_quantile','azunet_local','unet_raw','unet_quantile','unet_local']
    xs=np.arange(len(selected));width=.34
    for j,(direction,section) in enumerate(results['directions'].items()):
        vals=[section['methods'][key]['mean_AUROC'] for key in selected]
        ax.bar(xs+(j-.5)*width,vals,width,label=direction.replace('_',' '))
    ax.axhline(.5,linestyle='--',linewidth=1)
    ax.set_ylim(0,.85);ax.set_xticks(xs,labels=['AZ raw','AZ quantile','AZ local','UNet raw','UNet quantile','UNet local'])
    ax.set_ylabel('Mean AUROC: injected artificial track vs artificial scratch')
    ax.set_title('S33: held-out H50Z source photo, two directions (one physical run)')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output_dir/'S33_cross_photo_AUROC.png',dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.9,4.2))
    seeds=results['S34_axial']['new_unseen_injection_seeds']
    values=results['S34_axial']['delta_AUROC_per_seed']
    ax.bar([str(x) for x in seeds],values)
    ax.axhline(0,linestyle='--',linewidth=1)
    ax.set_title('S34: machine-axis prior does not consistently help AZUNet')
    ax.set_ylabel('AUROC(axial loss) − AUROC(baseline local AZUNet)')
    ax.set_xlabel('Reserved synthetic insertion seed (H50Z-006)')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output_dir/'S34_axis_ablation_delta.png',dpi=165);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 x=compile_results(a.results)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
 make_charts(x,a.out.parent)
 print('REPORT',a.out,'forward',x['directions']['005_train_006_test']['methods']['azunet_quantile'],
       'reverse',x['directions']['006_train_005_test']['methods']['unet_local'])
