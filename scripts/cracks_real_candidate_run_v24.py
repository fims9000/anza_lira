"""V24 post-V23 exploratory candidate-recall repair on genuine CRACKS pixels.

Reuse the exact V23 crowd-only training split and HGB front-end. Compare
unmodified V23 controls vs 68px terminal candidates and terminal-to-visible
interior landing ports, with expert masks read only after inference.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import json
import zipfile

import numpy as np

from scripts.cracks_real_auto_core_v23 import anisotropic_features,run_pipeline
from scripts.cracks_real_candidate_v24 import run_expanded
from scripts.cracks_real_auto_run_v23 import (
    load_image,load_mask,zip_index,_train,
    evaluate_mask,candidate_reachability,_bootstrap,sha_file,_overlay
)

METHODS=('raw','context','geometry','terminal_extended','virtual_landing')


def run(images_path:Path,annotations_path:Path,out:Path,*,max_expert=40):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(images_path) as images_zip, zipfile.ZipFile(annotations_path) as ann_zip:
        images,annotations=zip_index(images_zip,ann_zip)
        expert_ids=sorted(i for i in set(images)&set(annotations) if
                          'expert' in annotations[i])[:max_expert]
        if len(expert_ids)<10:raise RuntimeError('Insufficient expert sections')
        model,threshold,receipt=_train(images_zip,ann_zip,images,annotations,out=out)
        rows=[]
        for k,sid in enumerate(expert_ids):
            image=load_image(images_zip,images[sid])
            features=anisotropic_features(image)
            prob=model.predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(image.shape).astype(np.float32)
            predictions={
                'raw':run_pipeline(prob,threshold=threshold,strategy='context'),
                'context':run_pipeline(prob,threshold=threshold,strategy='context'),
                'geometry':run_pipeline(prob,threshold=threshold,strategy='geometry'),
                'terminal_extended':run_expanded(prob,threshold,mode='terminal_extended'),
                'virtual_landing':run_expanded(prob,threshold,mode='virtual_landing'),
            }
            # Expert annotations are read ONLY after all model choices.
            expert=load_mask(ann_zip,annotations[sid]['expert'])
            row={'section':sid,'split':'V24_POST_V23_EXPLORATORY'}
            for name in METHODS:
                predicted=predictions[name]
                mask=predicted['raw_mask'] if name=='raw' else predicted['completed_mask']
                metrics=evaluate_mask(mask,expert)
                audit=candidate_reachability(predicted,expert)
                for key,value in metrics.items():row[name+'_'+key]=value
                for key,value in audit.items():row[name+'_candidate_'+key]=value
                row[name+'_accepted_links']=0 if name=='raw' else len(predicted['selected'])
                row[name+'_no_link_ports']=len(predicted['ports']) if name=='raw' else predicted['none_count']
            rows.append(row)
            if k<4:
                _overlay(image,predictions['raw']['raw_mask'],
                         predictions['virtual_landing']['completed_mask'],
                         out/f'v24_virtual_section_{sid:03}.png')
                np.savez_compressed(out/f'v24_predict_section_{sid:03}.npz',
                   probability=prob.astype(np.float16),
                   raw_mask=predictions['raw']['raw_mask'].astype(np.uint8),
                   extended_mask=predictions['terminal_extended']['completed_mask'].astype(np.uint8),
                   virtual_mask=predictions['virtual_landing']['completed_mask'].astype(np.uint8))
            print('V24 EVAL',sid,'connected',[(m,row[m+'_connected_gold_anchor_pairs'])
                               for m in METHODS],
                  'candidate_direct',[(m,row[m+'_candidate_direct_candidate_could_connect'])
                               for m in METHODS],flush=True)
    if not rows:raise RuntimeError('No evaluated sections')
    with (out/'per_section_v24.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    aggregate={}
    for name in METHODS:
        positives=sum(r[name+'_N_expert_blue'] for r in rows)
        negatives=sum(r[name+'_N_explicit_orange'] for r in rows)
        hit=sum(r[name+'_expert_blue_recalled_tol2'] for r in rows)
        fp=sum(r[name+'_explicit_orange_false_pixels'] for r in rows)
        denom=sum(r[name+'_eligible_paths'] for r in rows)
        connected=sum(r[name+'_connected_gold_anchor_pairs'] for r in rows)
        misses=sum(r[name+'_candidate_unconnected_but_both_visible'] for r in rows)
        reach=sum(r[name+'_candidate_direct_candidate_could_connect'] for r in rows)
        chain=sum(r[name+'_candidate_candidate_chain_could_connect'] for r in rows)
        aggregate[name]={
            'blue_recall_tol2':hit/positives if positives else None,
            'false_fraction_explicit_orange':fp/negatives if negatives else None,
            'N_explicit_blue':positives,'N_explicit_orange':negatives,
            'N_expert_annotated_paths':denom,
            'connected_expert_anchor_pairs':connected,
            'connectivity_fraction':connected/denom if denom else None,
            'unconnected_with_both_anchors':misses,
            'direct_candidate_reachable':reach,
            'any_candidate_chain_reachable':chain,
            'num_all_candidates':sum(r[name+'_candidate_candidate_count'] for r in rows),
            'accepted_links':sum(r[name+'_accepted_links'] for r in rows),
        }
    report={
        'protocol':'CRACKS_V24_EXPLORATORY_AUTONOMOUS_CANDIDATE_RECALL_REPAIR',
        'status':'EXECUTED_EXPLORATORY',
        'method_is_original_pretrained_ANZA_LIRA':False,
        'this_protocol_was_designed_after_opening_V23_expert_results':True,
        'uses_oracle_endpoints_for_inference':False,
        'model_trained_only_nonexpert_certain_blue_orange':True,
        'expert_masks_read_only_post_prediction':True,
        'independent_geological_fault_identity_labels':False,
        'all_sections_one_F3_volume':True,
        'training':receipt,
        'sections':len(rows),
        'methods':aggregate,
        'virtual_vs_context_connectivity_section_bootstrap':_bootstrap(rows,'virtual_landing','context'),
        'extended_vs_context_connectivity_section_bootstrap':_bootstrap(rows,'terminal_extended','context'),
        'no_new_calibration_used_for_V24_decisions':True,
        'per_section_csv_sha256':sha_file(out/'per_section_v24.csv'),
        'source_images_zip_sha256':sha_file(images_path),
        'source_annotations_zip_sha256':sha_file(annotations_path),
        'scientific_boundary':'V24 is post-V23 exploratory; candidate upper bound uses expert anchors for analysis only, not link selection. Expert masks have semantic pixels, not geological instance IDs.'
    }
    (out/'report_v24.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'methods':aggregate},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('results/cracks_real_candidate_v24'))
    p.add_argument('--max-expert',type=int,default=40)
    a=p.parse_args()
    run(a.images,a.annotations,a.out,max_expert=a.max_expert)

if __name__=='__main__':main()
