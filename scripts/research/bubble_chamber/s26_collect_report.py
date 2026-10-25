#!/usr/bin/env python3
"""Frozen S26 synthesis, no posthoc re-selection on heldout photo 006."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/data/s26_work/results')
rows={}
for seed in (20261027,20261028):
    d=json.load(open(ROOT/f'seed{seed}_full/experiment.json'))
    for key,v in d['methods'].items():
        rows.setdefault(key,{})[str(seed)]=v['heldout_006']
    guided=ROOT/(f'guided_seed{seed}_8')
    for model in ('unet','azunet'):
        if (guided/model/'experiment.json').is_file():
            x=json.load(open(guided/model/'experiment.json'))
            rows.setdefault(f'{model}_two_channel',{})[str(seed)]=x['heldout_006']
    if seed==20261027:
        x=json.load(open(ROOT/'guided_seed20261027/azunet/partial_eval.json'))
        rows.setdefault('azunet_two_channel',{})[str(seed)]=x['test_006']
    hand=json.load(open(ROOT/f'handcrafted_seed{seed}.json'))
    for method,v in hand['results'].items():
        rows.setdefault(method,{})[str(seed)]=v['heldout_006']
paired=json.load(open(ROOT/'paired_patch_intervals.json'))
anchors=json.load(open(ROOT/'real_anchor_baselines.json'))
frags=json.load(open(ROOT/'real_geometry/all_crops_summary.json'))
report={'schema':'s26_two_real_photo_research_summary_v1','date':'2026-10-25',
  'reference_qualifiers':{
    'real_h50z_photos':2,'photo_ids':['005','006'],'same_physical_run':True,
    'native_expert_pixel_track_masks':0,'native_expert_track_pair_links':0,
    'source_visualmetro_tracks_used_as_synthetic_template':True,
    'heldout_006_used_for_model_training_or_threshold_tuning':False,
    'repeated_exploratory_model_comparison_on_006':True,
    'conclusion_for_independent_photo_run':'NOT_YET_VALIDATED',
  },'scores_synthetic_injected_tracks_vs_synthetic_scratches':rows,
  'conditional_patch_bootstrap':paired,
  'independent_target_real_anchor_start_vs_fiducial':anchors,
  'authentic_photo_fragment_proposals_UNVERIFIED':frags,
  'physical_join_auto_merge_count':0}
(ROOT/'S26_FINAL_RESEARCH_RECEIPT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
order=['sato','blackhat','steerable_ridge','multi_feature_HGB','unet_raw','unet_local','azunet_raw','azunet_local','unet_two_channel','azunet_two_channel']
labels=['Sato','Black hat','Steerable','HGB 7 features','U-Net raw','U-Net local','AZUNet raw','AZUNet local','U-Net + ridge','AZUNet + ridge']
fig,ax=plt.subplots(figsize=(12,6))
xs=np.arange(len(order));w=.38
for i,seed in enumerate((20261027,20261028)):
    vals=[rows.get(k,{}).get(str(seed),{}).get('roc_auc_synthetic_marks',np.nan) for k in order]
    ax.bar(xs+(i-.5)*w,vals,w,label=f'photo 006: injection seed {seed}')
ax.set_xticks(xs,labels,rotation=35,ha='right')
ax.set_ylabel('ROC AUC on synthetic track vs scratch pixels')
ax.set_ylim(0,1)
ax.set_title('S26 H50Z true photographic background, synthetic overlays only\nFit + calibration: photo 005  •  test: photo 006 (same physical run)')
ax.legend();ax.grid(axis='y',alpha=.22);fig.tight_layout()
fig.savefig(ROOT/'S26_method_comparison.png',dpi=175);plt.close(fig)
print('methods',len(rows),'real fragments',sum(x['segments'] for x in frags['source_crops']),
      'candidate joins',sum(x['possible_links_review_only'] for x in frags['source_crops']))
for k in order:
 print(k, {seed:round(rows.get(k,{}).get(str(seed),{}).get('roc_auc_synthetic_marks',float('nan')),4) for seed in (20261027,20261028)})
