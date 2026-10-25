#!/usr/bin/env python3
"""Merge trusted source-locked partial AZUNet evidence; never create human GT."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from bubble_chamber.s33_normalized_inference import SCHEMA


def merge(drafts,parts):
    src=json.loads(Path(drafts).read_text())
    if src.get('schema')!='s31_machine_first_h50z_draft_v1':raise ValueError('Original source pack required')
    indexed={d['draft_id']:(rec['crop_id'],rec['photo_id']) for rec in src['photos'] for d in rec['drafts']}
    rows=[];seen=set();chunks=[]
    for path in parts:
        chunk=json.loads(Path(path).read_text())
        if chunk.get('schema')!=SCHEMA or chunk.get('physics_auto_join_authorized') is not False:
            raise ValueError('Invalid chunk or physical auto-join claim')
        chunks.append(Path(path).name)
        for r in chunk['rows']:
            id=r['draft_id']
            if id in seen or id not in indexed or indexed[id]!=(r['crop_id'],r['source_photo']):
                raise ValueError('Duplicated/foreign candidate evidence')
            if r['expert_approved'] or r['physics_auto_merge_allowed'] or r['machine_draft_remains']!='MACHINE_DRAFT_UNREVIEWED':
                raise ValueError('Unverified candidate promoted to expert')
            seen.add(id);rows.append(r)
    if seen!=set(indexed):raise ValueError('Missing proposals or unrecognized source snapshot')
    samples={mode:np.asarray([r['AZUNet_'+mode].get('median_center_minus_flank',np.nan) for r in rows],float)
             for mode in ('quantile','local')}
    stats={}
    for mode,a in samples.items():
        finite=a[np.isfinite(a)]
        stats[mode]={'with_finite_centerline_evidence':int(len(finite)),
           'positive_center_minus_flank_count':int((finite>0).sum()),
           'median_center_minus_flank':float(np.median(finite)),
           'q10_q90':np.quantile(finite,[.1,.9]).tolist()}
    result={'schema':SCHEMA,'source_crops':len(src['photos']),
        'original_draft_count':len(indexed),'merged_reviewer_drafts':len(rows),
        'input_chunks':chunks,'photos':['005','006'],'physics_auto_join_authorized':False,
        'true_expert_H50Z_track_labels_added':0,'machine_likelihoods_are_calibrated':False,
        'review_recommendation':'Check source image, line curvature and competing paths; model evidence unverified',
        'summary':stats,'rows':rows}
    return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--drafts',type=Path,required=True)
 p.add_argument('--parts',nargs='+',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 out=merge(a.drafts,a.parts)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print('MERGED',out['merged_reviewer_drafts'],'evidence only; per-model',out['summary'])
