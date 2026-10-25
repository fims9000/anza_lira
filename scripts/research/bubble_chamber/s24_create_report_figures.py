#!/usr/bin/env python3
"""Create figures solely from an existing immutable S24 metric receipt."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=json.loads(args.metrics.read_text())
    if r['schema']!='S24_CONSENSUS_SELECTIVE_AUDIT_V1':raise ValueError('wrong metric receipt')
    names=['Ordinary geometry HGB','Graph-context HGB','Consensus']
    keys=['geometry_hgb','graph_context_hgb','consensus']
    tp=[sum(x[k]['tp'] for x in r['results']) for k in keys]
    fp=[sum(x[k]['fp'] for x in r['results']) for k in keys]
    fig,ax=plt.subplots(figsize=(8.2,4.9))
    pos=np.arange(len(names));a=ax.bar(pos-.18,tp,width=.36,label='Correct links (TP)')
    b=ax.bar(pos+.18,fp,width=.36,label='Wrong links (FP)')
    ax.bar_label(a);ax.bar_label(b)
    ax.set_xticks(pos,names);ax.set_ylabel('Aggregated links across 3 same-film splits')
    ax.set_title('S24: Lower false-join count costs correct continuations')
    ax.legend();fig.tight_layout()
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/'s24_correct_wrong_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    levels=list(r['registered_config']['review_budgets'])
    smart=[sum(x['review_simulation']['uncertainty_diversity_queue'][str(b)]['manually_verified_correct_joins_added'] for x in r['results']) for b in levels]
    baseline=[sum(x['review_simulation']['hash_control'][str(b)]['manually_verified_correct_joins_added'] for x in r['results']) for b in levels]
    ax.plot(levels,smart,marker='o',label='Disagreement / frame-diverse review')
    ax.plot(levels,baseline,marker='s',label='Hash-ordered review control')
    ax.set_xlabel('Additional simulated reviews per split')
    ax.set_ylabel('Expert-confirmed correct links added across three splits')
    ax.set_title('S24: Ranking has not improved over simple review control')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(args.out/'s24_simulated_review_budget.png',dpi=160);plt.close(fig)
    print('FIGURES',*[str(x) for x in args.out.glob('*.png')])


if __name__=='__main__':main()
