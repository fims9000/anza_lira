#!/usr/bin/env python3
"""Collect frozen S21/S22 per-split outputs without reselecting model on tests."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pooled(rows,key):
    m=[r[key] for r in rows]
    t=sum(x['tp'] for x in m);f=sum(x['fp'] for x in m)
    positives=sum(x['positive_candidates'] for x in m)
    possible=sum(x['true_continuations_possible'] for x in m)
    return {'tp':t,'fp':f,'positive_candidates':positives,
            'true_possible':possible,'recall':round(t/max(positives,1),6),
            'precision_on_selected':round(t/max(t+f,1),6),
            'per_seed_tp_fp':[(r['split_seed'],r[key]['tp'],r[key]['fp']) for r in rows]}

def collect(s21_files,s22_file,outdir):
    a=[json.loads(Path(f).read_text()) for f in s21_files]
    b=json.loads(Path(s22_file).read_text())
    all21=[r for d in a for r in d['results']]
    summary={'schema':'S21_S22_FROZEN_REPORT_V1',
       'dataset_sha256':a[0]['source_sha256'],
       'observed_data':'real film-070 expert point coordinates; artificially withheld clicks, NOT genuine image gap masks',
       's21':{},'s22':{},
       'independent_film_validation':False,'real_photo_dense_labels':False,
       'unattended_physics_allowed':False}
    for budget in (160,320,640):
        for gap in (1,3,5):
            rows=[r for r in all21 if r['training_labels']==budget and r['tested_missing_clicks']==gap]
            summary['s21'][f'{budget}_gap{gap}']={k:pooled(rows,k) for k in ('baseline','context_logreg','graph_context')}
    for budget in (320,640):
        for gap in (3,5):
            rows=[r for r in b['results'] if r['training_label_budget']==budget and r['test_hidden_clicks']==gap]
            summary['s22'][f'{budget}_gap{gap}']={k:pooled(rows,k) for k in ('short_graph','mixed_graph','short_generic_hgb','mixed_generic_hgb')}
    Path(outdir).mkdir(parents=True,exist_ok=True)
    path=Path(outdir)/'s21_s22_summary.json'
    path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    budgets=[320,640]
    for name,label in [('short_graph','Graph context: short gaps only'),('mixed_graph','Graph context: mixed gaps'),
                       ('short_generic_hgb','Geometry HGB: short gaps'),('mixed_generic_hgb','Geometry HGB: mixed gaps')]:
        values=[summary['s22'][f'{b}_gap5'][name]['recall']*100 for b in budgets]
        ax.plot(budgets,values,'o-',label=label)
    ax.set_xlabel('Human-reviewed training pair decisions (validation adds 160)')
    ax.set_ylabel('Correct continuation recall on 5 hidden clicks (%)')
    ax.set_title('S22: held-out gap shift on measured visualMetro geometry\nSynthetic gaps, 3 overlapping frame splits of film 070')
    ax.set_xticks(budgets)
    ax.set_ylim(0,85)
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(Path(outdir)/'s22_gap_shift_recall.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,5.2))
    for name,label in [('short_graph','Graph context short'),('mixed_graph','Graph context mixed'),
                       ('short_generic_hgb','Geometry short'),('mixed_generic_hgb','Geometry mixed')]:
        values=[summary['s22'][f'{b}_gap5'][name]['fp'] for b in budgets]
        ax.plot(budgets,values,'o-',label=label)
    ax.set_xlabel('Human-reviewed training pair decisions (+160 calibration)')
    ax.set_ylabel('False joins on test candidates (3 splits combined)')
    ax.set_title('S22: false links remain at nonzero recall\nDo not authorize physical auto-joining')
    ax.set_xticks(budgets);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(Path(outdir)/'s22_false_links.png',dpi=180);plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s21',nargs=3,type=Path,required=True)
    p.add_argument('--s22',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    s=collect(a.s21,a.s22,a.out_dir)
    for b in (320,640):
        d=s['s22'][f'{b}_gap5']
        print('BUDGET',b,'short',d['short_graph'],'mixed',d['mixed_graph'])
