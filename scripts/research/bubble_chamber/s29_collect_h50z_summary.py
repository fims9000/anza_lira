#!/usr/bin/env python3
"""Summarize source-locked S27/S28/S29 reports without ever recasting synthetic labels as real GT."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def collect(overlap:Path,synthetic:Path,s28:Path,s29:Path,physical_metrics:Path,out:Path):
    a=json.loads(overlap.read_text())
    b=json.loads(synthetic.read_text())
    c=json.loads(s28.read_text())
    d=json.loads(s29.read_text())
    e=json.loads(physical_metrics.read_text())
    assert a['source_crops']==39 and len(b['trials'])==16 and c['n_exploratory_trials']==16
    assert d['number_of_software_injected_trials']==8 and len(e['rows'])==72
    assert a['human_pixel_track_masks']==0 and c['real_native_track_ground_truth_labels']==0
    mean_auc={}
    for model in ('unet','azunet','guided_az'):
        mean_auc[model]={part:float(np.mean([r['methods'][model][part]['synthetic_roc_auc'] for r in b['trials']])) for part in ('stride64','stride32')}
    report={'schema':'S27_S29_TRUTH_DISTINCTIONS_AND_REPRODUCIBLE_METRICS_V1',
            'study_date':'2026-10-25',
            'photographs':{'H50Z_030_01_005':21,'H50Z_030_01_006':18,'physical_run_count':1},
            'real_operator_start_points':9,
            'real_expert_contour_track_masks':0,
            'real_expert_same_track_relation_labels':0,
            'S27_seam_absolute_predicted_probability_jump':a['source_photos'],
            'S27_long_injected_synthetic_AUC':mean_auc,
            'S28_injected_known_anchor_prefix_recovery':c['summary'],
            'S29_anchor_offset_plus_erased_synthetic_gaps':d['summary'],
            'S29_real_photo_candidate_provisional_measurements':len(e['rows']),
            'S29_physical_particle_identity_confirmations':0,
            'source_original_weights_sha256':a['source_model_checkpoint_sha256'],
            'preexisting_source_S26_thresholds':a['frozen_thresholds_calibrated_only_on_SYNTHETIC_photo005_marks'],
            'strict_evidence':{
              'pixel_probabilities_seam_statistic_is_not_true_label_accuracy':True,
              'artificial_injected_traces_are_the_ONLY_ground_truth_in_all_tests':True,
              'S28_perfect_known_seed_is_not_real_operator_localization_validation':True,
              'S29_seed_error_is_software_added_not_measured':True,
              'oracle_best_of_top4_never_operator_usable':True,
              'real_native_particle_2d_curve_is_undetermined':True,
              '3D_particle_momentum_or_type_is_unavailable':True,
              'automatic_physics_merges_must_remain_disabled':True,
            }}
    out.mkdir(parents=True,exist_ok=True)
    (out/'s27_s29_public_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    fig,ax=plt.subplots(figsize=(9,4.9))
    data=['H50Z-005','H50Z-006']
    for model,label in [('unet','U-Net'),('azunet','Original AZUNet'),('guided_az','Guided AZUNet')]:
        vals=[a['source_photos'][p][model]['pooled_boundary_jump_reduction_pct'] for p in ('005','006')]
        ax.plot(data,vals,'o-',label=label)
    ax.set_title('S27: reduction of artificial seam jump on 39 real H50Z crops')
    ax.set_ylabel('Relative reduction of boundary artifact (%)')
    ax.set_ylim(0,100);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'S27_boundary_artifact_reduction.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.9))
    models=['straight_evidence','curvature_constrained_evidence','original_AZ_plus_curvature']
    vals=[c['summary'][name]['top1']['mean_injected_prefix_coverage']*100 for name in models]
    ax.bar(['Straight','Curvature + photometry','Curvature + original ANZA'],vals)
    ax.set_title('S28: injected known-start trace coverage on native H50Z background\nSynthetic track truth only; 16 trials')
    ax.set_ylabel('Mean injected prefix recovered within 3px (%)')
    ax.set_ylim(0,100);ax.grid(alpha=.2,axis='y')
    fig.tight_layout();fig.savefig(out/'S28_anchored_synthetic_path_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.9))
    for gap in (False,True):
        values=[d['summary'][f'gap{int(gap)}_offset{j}_top1']['mean_prefix_coverage']*100 for j in (0,3,6,10)]
        ax.plot([0,3,6,10],values,'o-',label='Two artificial erasures' if gap else 'No erased segments')
    ax.set_title('S29: robustness to synthetic initial-point error\n8 new overlays on real H50Z-006 backgrounds, not real GT')
    ax.set_ylabel('Mean injected prefix recovered within 3px (%)')
    ax.set_xlabel('Artificial vertical error in starting point (pixels)')
    ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'S29_seed_jitter_stress.png',dpi=150);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ('overlap','synthetic','s28','s29','physical_metrics','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=p.parse_args();report=collect(args.overlap,args.synthetic,args.s28,args.s29,args.physical_metrics,args.out)
    print('S27-S29 safe summary generated',len(report['S29_anchor_offset_plus_erased_synthetic_gaps']))
if __name__=='__main__':main()
