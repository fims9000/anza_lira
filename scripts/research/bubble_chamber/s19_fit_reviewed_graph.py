#!/usr/bin/env python3
"""Train Graph-LIRA only on provenance-bound, explicitly reviewed real-photo pairs.

At least 10 reviewed same/different pairs with both labels are required.
Output is strictly provisional while independent-run calibration is absent.
No pairs/negative labels are derived from geometric proximity or unmarked pixels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s14_fewshot_graph import fit_fewshot,json_model_parameters,features_from_segment_pair


def fit_from_real_pair_file(path:Path,out_dir:Path):
    source=json.loads(Path(path).read_text(encoding='utf-8'))
    if source.get('schema')!='s19_expert_pair_labels_v1' or source.get('independent_run_calibrated') is not False:
        raise ValueError('unrecognized or falsely calibrated real-photo review source')
    rows=source.get('rows')
    if not isinstance(rows,list):raise ValueError('review rows missing')
    for r in rows:
        if r.get('schema')!='s19_expert_track_pair_v1' or r.get('verified_human') is not True:
            raise ValueError('pair is not from an explicitly reviewed expert decision')
        if r.get('expert_label') not in {'same_track','different_track'}:
            raise ValueError('unreviewed pairs must never enter Graph-LIRA training')
        if r.get('orientation')!='minimum_endpoint_distance_unverified':
            raise ValueError('unexpected orientation provenance')
        features_from_segment_pair(r)
    positive=sum(r['expert_label']=='same_track' for r in rows)
    negative=len(rows)-positive
    runs=sorted(set(r['run_id'] for r in rows))
    status=('TRAINABLE_PROVISIONAL_ONLY' if len(rows)>=10 and positive>0 and negative>0 and len(runs)==1
            else 'INSUFFICIENT_EXPERT_RELATION_LABELS_ABSTAIN')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    report={'schema':'s19_sparse_real_pair_graph_training_receipt_v1',
            'status':status,'human_confirmed_pairs':len(rows),'same_track':positive,
            'different_track':negative,'source_runs':runs,
            'independent_real_run_validation':False,
            'real_automatic_track_merges_allowed':False,
            'source_package_sha256':source.get('source_package_sha256'),
            'limitations':['Two H50Z source BMPs are from the same physical run',
                           'Unmarked pairs are unknown, never negative training data',
                           'No independent run or physically certified false-join calibration']}
    if status=='TRAINABLE_PROVISIONAL_ONLY':
        model,stats=fit_fewshot(rows,source_run_id=runs[0])
        (out/'s19_provisional_graph_model.json').write_text(json.dumps(json_model_parameters(model),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report['training']=stats
    (out/'s19_graph_fit_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expert-pairs',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    report=fit_from_real_pair_file(a.expert_pairs,a.out_dir)
    print(json.dumps({k:v for k,v in report.items() if k not in {'limitations','training'}},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
