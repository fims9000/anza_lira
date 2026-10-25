#!/usr/bin/env python3
"""Collect frozen S36 result, explicitly separate post-hoc decomposition."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bubble_chamber.s36_saturation_normalization import S36_TEST_FORWARD_SEEDS,S36_TEST_REVERSE_SEEDS


def summarize(paths,posthoc,outdir):
    results={name:json.loads(Path(f).read_text()) for name,f in paths.items()}
    post={name:json.loads(Path(f).read_text()) for name,f in posthoc.items()}
    for photo,obj in results.items():
        seeds=S36_TEST_FORWARD_SEEDS if photo=='005' else S36_TEST_REVERSE_SEEDS
        if (obj.get('schema')!='s36_saturation_aware_photo_transfer_synthetic_overlay_only_v1' or
            obj.get('source_train_photo')!=photo or obj.get('test_seeds')!=list(seeds) or
            obj.get('trained_on_human_certified_tracks') is not False or len(obj['results'])!=16):
            raise ValueError('Unexpected S36 frozen test or physical GT source')
        if post[photo].get('EXPLORATORY_POST_TEST_NOT_INDEPENDENT_CONFIRMATION') is not True:
            raise ValueError('Explanatory decomposition cannot be reported as preregistered')
    summary={'schema':'s36_predeclared_saturation_blend_summary_v1',
        'source_material':'one physical run, two original H50Z photos, machine S31 marks, artificial test tracks+scratches',
        'real_native_track_ground_truth':False,'human_approved_pair_truth':False,
        'independent_physical_runs':0,'automatic_physical_track_joins_authorized':False,
        'test_heldout_injection_seeds':{'005_to_006':list(S36_TEST_FORWARD_SEEDS),'006_to_005':list(S36_TEST_REVERSE_SEEDS)},
        'new_injection_scenes_per_direction':192,'extra_exposure_variants_per_case':4,
        'normalizer':'z = (1-alpha) * quantile(I) + alpha * local(I); alpha = clip((fraction(I<=1 or I>=254)-.02)/.18,0,1)',
        'test_results':{},'posthoc_source_of_effect_not_confirmatory':{},
        'selection_policy':'no tuning or checkpoint selection on either target photograph or these S36 injected labels',
        'recommendation':'Do not enable saturation blend globally without independent run validation; apparent gain is direction dependent and mostly from training perturbation, not inference blend.'}
    for direction,source in [('005_to_006','005'),('006_to_005','006')]:
        report=results[source];summary['test_results'][direction]={}
        for mode in ('identity','bright','dark','highlight_clip'):
            rec=[x for x in report['results'] if x['perturbation']==mode]
            names=('AZUNet_quantile_S35_matched','AZUNet_saturation_blend_S36','U_Net_local_S35_matched')
            vec={n:np.asarray([x['metrics'][n]['synthetic_injected_only_AUROC'] for x in rec]) for n in names}
            delta=vec[names[1]]-vec[names[0]]
            gen=np.random.default_rng(20261025)
            boot=delta[gen.integers(0,4,size=(10000,4))].mean(1)
            summary['test_results'][direction][mode]={
                'n_injected_scenes':sum(x['artificial_scenes'] for x in rec),
                'mean_source_clipped_pixels_fraction':float(np.mean([x['mean_input_clipped_fraction'] for x in rec])),
                'mean_local_blend_fraction_UNCALIBRATED':float(np.mean([x['mean_saturation_blend_local_weight'] for x in rec])),
                'mean_AUROC_generated_only':{n:float(a.mean()) for n,a in vec.items()},
                'paired_delta_s36_minus_s35_AZUNet_mean':float(delta.mean()),
                'paired_delta_by_synthetic_seed':delta.tolist(),
                'seed_delta_bootstrap_95CI_NOT_INDEPENDENT_RUN':[float(v) for v in np.quantile(boot,[.025,.975])],
                'number_of_seeds_improved_AZUNet':int((delta>0).sum())}
        post_obj=post[source]
        summary['posthoc_source_of_effect_not_confirmatory'][direction]={}
        for mode in ('identity','highlight_clip'):
            rows=[r for r in post_obj['results'] if r['mode']==mode]
            summary['posthoc_source_of_effect_not_confirmatory'][direction][mode]={
                key:float(np.mean([r[key] for r in rows])) for key in ('delta_inference_only','delta_training_only','delta_full')}
    folder=Path(outdir);folder.mkdir(parents=True,exist_ok=True)
    (folder/'S36_PUBLIC_FROZEN_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,5));labels=['005 → 006','006 → 005'];x=np.arange(len(labels));w=.25
    names=('AZUNet_quantile_S35_matched','AZUNet_saturation_blend_S36','U_Net_local_S35_matched')
    for i,(n,label) in enumerate(zip(names,('AZUNet quantile (S35)','AZUNet saturation blend (S36)','U-Net local control'))):
        y=[summary['test_results'][d]['identity']['mean_AUROC_generated_only'][n] for d in ('005_to_006','006_to_005')]
        ax.bar(x+(i-1)*w,y,w,label=label)
    ax.set_xticks(x,labels);ax.set_ylabel('AUROC: injected track vs injected scratch only')
    ax.set_ylim(.50,.76);ax.set_title('S36 cross-photo transfer depends on training direction')
    ax.legend(fontsize=9);ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(folder/'S36_cross_photo_directions.png',dpi=175);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));modes=('identity','bright','dark','highlight_clip');positions=np.arange(len(modes))
    for d,l in (('005_to_006','005 → 006'),('006_to_005','006 → 005')):
        points=[summary['test_results'][d][p]['paired_delta_s36_minus_s35_AZUNet_mean'] for p in modes]
        ax.plot(positions,points,'o-',label=l)
    ax.axhline(0,linestyle='--',linewidth=1);ax.set_xticks(positions,modes)
    ax.set_ylabel('S36 − S35 AZUNet AUROC on artificial controls')
    ax.set_title('S36: not a universally better exposure normalizer')
    ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'S36_perturbation_AUROC_delta.png',dpi=175);plt.close(fig)
    print(json.dumps({d:summary['test_results'][d]['identity'] for d in summary['test_results']},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('forward','reverse','posthoc-forward','posthoc-reverse','out-dir'):
        p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args()
    summarize({'005':a.forward,'006':a.reverse},{'005':a.posthoc_forward,'006':a.posthoc_reverse},a.out_dir)
