#!/usr/bin/env python3
"""Run Graph-LIRA after a SMALL number of P0 human pair reviews.

A review manifest lists original P0 results and human JSON decisions, with
explicit film/run/frame/split. All unknown candidates remain unknown. This
workflow only certifies outputs after sufficient reviewed independent-run
calibration; before then it produces tentative probabilities and abstains.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s14_fewshot_graph import (pair_rows_from_review,fit_fewshot,
                    calibrate_and_score,json_model_parameters,portable_predict)
from bubble_chamber.review import SCHEMA,result_digest


def load_review_manifest(path:Path):
    path=Path(path).resolve()
    raw=json.loads(path.read_text(encoding='utf-8'))
    if raw.get('schema')!='s14_p0_pair_review_manifest_v1':raise ValueError('unsupported review manifest schema')
    entries=raw.get('entries')
    if not isinstance(entries,list) or not entries:raise ValueError('empty P0 review manifest')
    collected=defaultdict(list);pairs_seen=set()
    for entry in entries:
        for key in ('film_id','run_id','frame_id','split','p0_result'):
            if not isinstance(entry.get(key),str) or not entry[key].strip():raise ValueError(f'missing {key}')
        split=entry['split']
        if split not in {'train','calibration','inference'}:raise ValueError('invalid stage')
        p=(path.parent/entry['p0_result']).resolve()
        if not p.is_relative_to(path.parent) or not p.is_file():raise ValueError('bad P0 source path')
        result=json.loads(p.read_text(encoding='utf-8'))
        if split=='inference':
            if entry.get('review_json') is not None:raise ValueError('unattended inference must not contain expert labels')
            # Empty decisions only as a *schema adapter*; no human labels are
            # synthesized and no automatic decisions are counted as reviews.
            review={'schema':SCHEMA,'source_sha256_grayscale_pixels':result['source_sha256_grayscale_pixels'],
                    'pipeline':result['pipeline'],'result_sha256':result_digest(result),'labels':[]}
        else:
            ref=entry.get('review_json')
            if not isinstance(ref,str) or not ref.strip():raise ValueError('review source mandatory')
            pr=(path.parent/ref).resolve()
            if not pr.is_relative_to(path.parent) or not pr.is_file():raise ValueError('bad review path')
            review=json.loads(pr.read_text(encoding='utf-8'))
        identity=(entry['run_id'],entry['frame_id'])
        if identity in pairs_seen:raise ValueError('duplicate frame in stage manifest')
        pairs_seen.add(identity)
        rows=pair_rows_from_review(result,review,film_id=entry['film_id'],
                      run_id=entry['run_id'],frame_id=entry['frame_id'])
        collected[split].extend(rows)
    if not collected['train'] or not collected['inference']:
        raise ValueError('training and inference pairs required')
    return dict(collected)


def run(path:Path,outdir:Path,*,source_run_id:str):
    groups=load_review_manifest(path)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    model,stat=fit_fewshot(groups['train'],source_run_id=source_run_id)
    parameters=json_model_parameters(model)
    (outdir/'relation_model.json').write_text(json.dumps(parameters,indent=2)+'\n')
    proposals=groups['inference']
    if groups.get('calibration'):
        result=calibrate_and_score(model,groups['calibration'],proposals,train_run_id=source_run_id)
    else:
        prediction=portable_predict(proposals,parameters)
        result={'schema':'s14_provisional_graph_lira_inference_v1',
                'status':'CALIBRATION_MISSING_ABSTAIN_ALL',
                'links':[{'scene_id':row['scene_id'],'left_segment':row['left_segment'],
                          'right_segment':row['right_segment'],
                          'probability_not_certainty':float(p),'proposed':False,
                          'reason':'no_independent_reviewed_calibration'} for row,p in zip(proposals,prediction)],
                'unattended_release_allowed':False}
    report={'schema':'s14_low_label_graph_review_receipt_v1','trained_on':stat,
            'unlabeled_foreground_background_assumptions':0,
            'counts':{key:len(rows) for key,rows in groups.items()},
            'inference':result,
            'user_images_uploaded_to_repository':False}
    (outdir/'inference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review-manifest',type=Path,required=True)
    p.add_argument('--train-run-id',required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    r=run(args.review_manifest,args.out_dir,source_run_id=args.train_run_id)
    print(json.dumps({'count':r['counts'],'unattended_release_allowed':r['inference']['unattended_release_allowed']},indent=2))
if __name__=='__main__':main()
