#!/usr/bin/env python3
"""Reconstruct an auditable PUBLIC summary of four private S35 experiments.

Raw H50Z pixels, original measured coordinates and pair IDs NEVER serialized
into the public JSON. The only labels are controlled artificial injections.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPLITS=(('005','006',20261121),('005','006',20261123),
        ('006','005',20261122),('006','005',20261124))


def aggregate(folder:Path):
    experiments=[]
    for train,test,seed in SPLITS:
        file=Path(folder)/f'S35_{train}_to_{test}_seed{seed}_SYNTHETIC_RESEARCH.json'
        rec=json.loads(file.read_text(encoding='utf8'))
        if rec.get('schema')!='s35_photo_and_frame_disjoint_synthetic_fragment_links_v1' or rec.get('photo_train')!=train or rec.get('photo_heldout')!=test or rec.get('seed_test')!=seed:
            raise ValueError('Missing/invalid preregistered H50Z test direction/seed')
        if rec['native_physical_track_GT']!=0 or rec['one_physical_run_only'] is not True or rec['thresholds_fit_on_train_photo_only'] is not True:
            raise ValueError('Accidental physics claim or test-photo leakage')
        if rec['test_n_scenes']!=40 or rec['train_n_scenes']!=40 or rec['calibration_n_scenes']!=24:
            raise ValueError('Unexpected sample budget')
        if rec['visualMetro_train_frames']!=108 or rec['visualMetro_heldout_frames']!=36:
            raise ValueError('Unexpected operator measurement frame split')
        experiments.append(rec)
    methods=list(experiments[0]['algorithms'])
    scores={}
    for arm in methods:
        tests=[e['algorithms'][arm]['heldout'] for e in experiments]
        correct=sum(t['correct_provisional_review_suggestions'] for t in tests)
        wrong=sum(t['wrong_provisional_review_suggestions'] for t in tests)
        scenes=sum(e['test_n_scenes'] for e in experiments)
        auc=[float(t['synthetic_only_AUROC']) for t in tests]
        scores[arm]={'mean_test_AUROC_synthetic_only':float(np.mean(auc)),
          'individual_AUROCs':[float(v) for v in auc],
          'test_AUROC_std_across_4_synthetic_bundles':float(np.std(auc)),
          'total_correct_provisional_links':int(correct),
          'total_wrong_provisional_links':int(wrong),
          'true_link_recall_under_abstention':float(correct/(2*scenes)),
          'wrong_proposed_links_per_100_scenes':float(wrong/scenes*100),
          'correct_and_wrong_not_physically_certified':True}
    n=sum(x['test_n_scenes'] for x in experiments)
    consensus={'correct':sum(x['geometry_and_ANZA_consensus']['correct'] for x in experiments),
       'wrong':sum(x['geometry_and_ANZA_consensus']['wrong'] for x in experiments),
       'review_only':True}
    assert n==160
    output={'schema':'s35_injected_fragment_link_reconstruction_public_report_v1',
        'date':'2026-10-25','two_authentic_H50Z_source_photographs':['005','006'],
        'independent_physical_run_count':1,'different_fill_or_B_field_test_available':False,
        'real_native_H50Z_expert_track_segment_labels':0,
        'operator_visualMetro_frame_train_count':108,'operator_visualMetro_frame_test_count':36,
        'machine_detector_centerlines_are_simulated_noisy':True,
        'source_3D_measurements_not_registered_to_H50Z_BMP':True,
        'synthetic_scenes_total':n,
        'synthetic_true_candidate_pairs':2*n,
        'synthetic_wrong_candidate_pairs':4*n,
        'per_direction_source_train_scenes':40,'per_direction_source_calibration_scenes':24,
        'photo_and_visualMetro_frame_disjoint_holdout':True,
        'photo_test_seeds_distinct_per_direction':[s for _,_,s in SPLITS],
        'threshold_calibration_max_empirical_wrong_pair_fraction':.02,
        'calibration_thresholds_not_real_world_safety_guarantees':True,
        'automatic_physics_links_enabled':False,
        'methods':scores,'geometry_AND_ANZA_consensus':consensus,
        'real_source_H50Z_S31_relations_scored':28,
        'real_H50Z_relations_certified_as_same_or_different':0,
        'preferred_for_manual_triage':'geometry+raw or geometry+local, compare ANZA disagreements; none are physics-authorized',
        'ANZA_experimental_effect':'S35 geometry+AZUNet improves synthetic recall but increases wrong proposals relative to geometric+raw; do NOT deploy automatic join',
        'limits':['All true/false labels artificial; original photo has unknown native tracks.',
                  'Two photos are from ONE run, no medium or field generalization measured.',
                  'Scene-level resampling not independent physical event statistics.',
                  'Source candidate centerline noise is synthetically injected, not measured detector error.',
                  'Empirical calibration FPR on synthetic source photo is not a bound on genuine physics errors.']}
    return output


def chart(summary,output:Path):
    title={'distance':'Nearest endpoint','geometry':'Geometry',
           'geometry+raw':'Geometry + raw','geometry+local':'Geometry + local',
           'geometry+AZUNet':'Geometry + AZUNet'}
    fig,ax=plt.subplots(figsize=(9.4,5.8))
    names=list(summary['methods'])
    xs=[summary['methods'][n]['total_wrong_provisional_links'] for n in names]
    ys=[summary['methods'][n]['total_correct_provisional_links'] for n in names]
    ax.scatter(xs,ys,s=90)
    for n,x,y in zip(names,xs,ys):ax.annotate(title[n],(x,y),xytext=(7,4),textcoords='offset points',fontsize=9)
    ax.set_xlim(-.5,max(xs)+4);ax.set_ylim(0,max(ys)+12)
    ax.set_xlabel('WRONG provisional joins / 160 synthetic scenes (lower is safer)')
    ax.set_ylabel('CORRECT provisional joins / 320 synthetic true pairs (higher is better)')
    ax.set_title('S35: selective link reconstruction on real-photo backgrounds (synthetic labels ONLY)')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--experiments',type=Path,required=True)
 p.add_argument('--json',type=Path,required=True);p.add_argument('--chart',type=Path,required=True)
 a=p.parse_args();summary=aggregate(a.experiments)
 a.json.parent.mkdir(parents=True,exist_ok=True)
 a.json.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 chart(summary,a.chart)
 print('S35 AGGREGATED',summary['synthetic_scenes_total'],
   'ANZA',summary['methods']['geometry+AZUNet']['total_correct_provisional_links'],
   summary['methods']['geometry+AZUNet']['total_wrong_provisional_links'])
