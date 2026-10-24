#!/usr/bin/env python3
"""S14 Graph-LIRA label-budget learning curve on audited visualMetro clicks."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s14_small_supervision import LabelBudgetProtocol,save_experiment

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261024)
    args=p.parse_args()
    manifest=audited_large_manifest(args.measurements)
    result=save_experiment(manifest,args.out_dir,LabelBudgetProtocol(seed=args.seed))
    print(json.dumps({'data':'source 070 manually measured track geometry; artificially hidden clicks',
       'review_budgets':result['protocol']['budgets'],
       'active_test':[{ 'budget':x['budget_labels'],
                       'generic_recall':x['models']['geometry_baseline']['test']['recall'],
                       'axial_recall':x['models']['axial_geometry_proxy']['test']['recall'],
                       'axial_false_connections':x['models']['axial_geometry_proxy']['test']['fp']}
                      for x in result['procedures']['balanced']],
       'limits':result['limitations']},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
