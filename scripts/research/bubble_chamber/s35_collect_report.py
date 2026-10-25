#!/usr/bin/env python3
"""Combine frozen S35 test receipts; no selection of model based on test seeds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bubble_chamber.s35_photometric_consistency import S35Protocol


def collect(forward,reverse,domain,train_forward,train_reverse,out_dir):
    cfg=S35Protocol();cfg.check()
    inputs={direction:json.loads(Path(path).read_text()) for direction,path in [('005_to_006',forward),('006_to_005',reverse)]}
    audit=json.loads(Path(domain).read_text())
    if audit.get('schema')!='s35_authentic_H50Z_photo_view_invariance_diagnostic_v1':raise ValueError('Wrong input diagnostic')
    for direction,obj in inputs.items():
        expected='005' if direction=='005_to_006' else '006'
        expected_seeds=cfg.test_forward_seeds if expected=='005' else cfg.test_reverse_seeds
        if (obj.get('schema')!='s35_cross_photo_machine_aug_consistency_synthetic_only_v1' or
            obj.get('train_photo')!=expected or obj.get('real_photo_track_ground_truth') is not False or
            len(obj.get('results',[]))!=12 or
            [r['test_seed'] for r in obj['results'] if r['perturbation']=='identity']!=list(expected_seeds)):
            raise ValueError('S35 test set/physical provenance changed')
    trains={name:json.loads(Path(path).read_text()) for name,path in [('005_to_006',train_forward),('006_to_005',train_reverse)]}
    for direction,t in trains.items():
        if (t.get('schema')!='s35_predeclared_photo_consistency_machine_only_v1' or
            t.get('trained_photo')!=inputs[direction]['train_photo'] or len(t.get('results',[]))!=8 or
            t.get('human_certified_track_labels')!=0):raise ValueError('Bad training receipt')
    report={'schema':'s35_photo_consistency_frozen_summary_v1',
        'source':'real H50Z backgrounds, S31 MACHINE weak labels and artificially injected track/scratch controls',
        'natural_real_bmp_track_gold_count':0,'operator_corrected_ground_truth':False,
        'both_photos_same_physical_run':True,'independent_physical_run':False,
        'epochs_fixed':cfg.epochs,'train_source_model_architectures_unchanged':True,
        'train_seeds':list(cfg.train_seeds),
        'test_seeds':{k:[r['test_seed'] for r in o['results'] if r['perturbation']=='identity'] for k,o in inputs.items()},
        'test_seed_repetitions_share_original_photographs_and_measured_templates':True,
        'test_n_injections_per_direction':192,'test_n_both_directions':384,
        'baseline_matched_two_photo_views_and_compute':True,
        'lambda_consistency':cfg.lambda_consistency,
        'domain_diagnostic':audit['per_photo'],
        'results':{},'fixed_research_conclusion':'No meaningful reliable improvement from extra unknown-pixel consistency; dominant remaining issue is noisy machine weak masks and saturation-driven photometric shift.',
        'physical_auto_join_authorized':False}
    out=Path(out_dir);out.mkdir(exist_ok=True,parents=True)
    details=[]
    for direction,obj in inputs.items():
        report['results'][direction]={}
        for arch in ('azunet','unet'):
            report['results'][direction][arch]={}
            for perturb in ('identity','bright','dark'):
                rows=[r for r in obj['results'] if r['perturbation']==perturb]
                baseline=np.asarray([r['metrics'][f'{arch}_matched_two_view']['AUROC_on_generated_track_vs_generated_scratch_pixels'] for r in rows])
                treatment=np.asarray([r['metrics'][f'{arch}_photo_consistency']['AUROC_on_generated_track_vs_generated_scratch_pixels'] for r in rows])
                delta=treatment-baseline
                rng=np.random.default_rng(20261025)
                # These 4 synthetic seed groups are NOT independent photos.
                bootstrap=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
                receipt={'n_injection_seeds':len(rows),'n_injected_scenes':sum(r['n_cases'] for r in rows),
                    'mean_AUROC_matched_two_view':float(baseline.mean()),
                    'mean_AUROC_extra_photo_consistency':float(treatment.mean()),
                    'mean_paired_delta_AUROC':float(delta.mean()),
                    'paired_seed_deltas':[float(q) for q in delta],
                    'cluster_seed_bootstrap_CI95_NOT_INDEPENDENT_RUNS':[float(q) for q in np.quantile(bootstrap,[.025,.975])],
                    'real_film_tracks_metric':'NOT_MEASURED'}
                report['results'][direction][arch][perturb]=receipt
                if perturb=='identity':
                    for i,r in enumerate(rows):
                        details.append({'direction':direction,'architecture':arch,'test_seed':r['test_seed'],
                                        'baseline_AUROC':float(baseline[i]),'consistency_AUROC':float(treatment[i]),
                                        'delta':float(delta[i])})
    for direction,t in trains.items():
        for arch in ('azunet','unet'):
            matched=[r for r in t['results'] if r['architecture']==arch and r['branch']=='photo_consistency']
            ratio=[]
            for r in matched:
                last=r['epochs'][-1]
                ratio.append((cfg.lambda_consistency*last['unknown_photometric_consistency_MSE'])/max(last['labelled_weak_supervised_BCE'],1e-8))
            report['results'][direction][arch]['last_epoch_unsupervised_to_supervised_magnitude_ratio_by_seed']=ratio
    (out/'S35_PUBLIC_AGGREGATED_METRICS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'S35_heldout_seed_deltas.csv').open('w') as f:
        f.write('direction,architecture,seed,baseline_AUROC,consistency_AUROC,delta\n')
        for row in details:
            f.write(f"{row['direction']},{row['architecture']},{row['test_seed']},{row['baseline_AUROC']:.8f},{row['consistency_AUROC']:.8f},{row['delta']:.8f}\n")
    # Plot of BEFORE vs AFTER, no invented native real-track Dice.
    fig,ax=plt.subplots(figsize=(9,5))
    for direction in ('005_to_006','006_to_005'):
        for arch in ('azunet','unet'):
            rec=report['results'][direction][arch]['identity']
            idx=len(ax.collections)
            ax.scatter(rec['mean_AUROC_matched_two_view'],rec['mean_AUROC_extra_photo_consistency'],
                       s=100,label=f'{direction}, {arch}')
    ax.plot([.5,.75],[.5,.75],linestyle='--',linewidth=1)
    ax.set(xlabel='Matched two-view baseline AUROC (synthetic only)',
           ylabel='Extra view-consistency AUROC (synthetic only)',
           title='S35: additional photometric consistency is nearly redundant')
    ax.set_xlim(.50,.75);ax.set_ylim(.50,.75);ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'S35_cross_photo_consistency_control.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    names=['raw','quantile','local'];positions=np.arange(len(names));width=.35
    for j,photo in enumerate(('005','006')):
        vals=[audit['per_photo'][photo]['mean_abs_normalized_input_change_under_seeded_affine_exposure'][k] for k in names]
        ax.bar(positions+(j-.5)*width,vals,width,label=f'H50Z photo {photo}')
    ax.set_xticks(positions,labels=names);ax.set_ylabel('Average absolute input shift, [0,1]')
    ax.set_title('S35: quantile invariance weakens on exposure-clipped photo 006')
    ax.legend();ax.grid(axis='y',alpha=.3)
    fig.tight_layout();fig.savefig(out/'S35_real_photo_exposure_shift.png',dpi=170);plt.close(fig)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('forward','reverse','domain','train-forward','train-reverse','out-dir'):
        p.add_argument('--'+n,required=True,type=Path)
    a=p.parse_args()
    r=collect(a.forward,a.reverse,a.domain,a.train_forward,a.train_reverse,a.out_dir)
    for d in ('005_to_006','006_to_005'):
        print(d,'AZ',r['results'][d]['azunet']['identity'])
