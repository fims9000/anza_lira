#!/usr/bin/env python3
"""Validate the private S31/S35 local review export without certifying physics.

Never publish input corrections, source IDs, photographs, centerline coordinates,
or individual answers. The optional summary has only aggregate counts and gates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import math

LINE_STATES={'USER_ACCEPTED_PROVISIONAL','USER_REJECTED_PROVISIONAL','USER_UNCERTAIN'}
PAIR_STATES={'LOCAL_SAME_PROVISIONAL','LOCAL_DIFFERENT_PROVISIONAL','LOCAL_UNSURE'}
GUARDED_KEYS=('schema','machine_source_s28_sha256','machine_source_s29_sha256',
             'all_unreviewed_stay_unknown','physical_joins_authorized','crops')


def validate_reviews(template, drafts, reviews):
    """Integrity/finite-geometry validation; *never* emits or confirms truth."""
    if any(reviews.get(k)!=template.get(k) for k in GUARDED_KEYS):
        raise ValueError('Source provenance/image crop mismatch or unsafe policy')
    if reviews['schema']!='s31_machine_corrections_v1' or not reviews['all_unreviewed_stay_unknown']:
        raise ValueError('Unknown correction format or unsafe UNKNOWN policy')
    if reviews['physical_joins_authorized'] is not False:
        raise ValueError('Physical merges cannot be approved by machine review')
    all_photos=drafts['photos']
    if len(all_photos)!=39: raise ValueError('Source inventory is not the expected 39 private crops')
    known_lines={d['draft_id']:d for p in all_photos for d in p['drafts']}
    known_pairs={r['relation_id']:r for p in all_photos for r in p['relation_drafts']}
    if len(known_lines)!=133 or len(known_pairs)!=28:
        raise ValueError('Missing source lines/pairs or duplicated IDs')
    lines=reviews.get('decisions');pairs=reviews.get('relation_decisions')
    if not isinstance(lines,dict) or not isinstance(pairs,dict):
        raise ValueError('Decisions must be maps')
    if any(k not in known_lines for k in lines) or any(k not in known_pairs for k in pairs):
        raise ValueError('Correction claims unknown source candidate')
    line_counts={k:0 for k in LINE_STATES}
    pair_counts={k:0 for k in PAIR_STATES}
    for k,v in lines.items():
        if not isinstance(v,dict) or v.get('status') not in LINE_STATES:
            raise ValueError('Invalid line decision')
        edits=v.get('edited_points_xy')
        if edits is not None:
            if not isinstance(edits,list) or len(edits)<2 or any(not isinstance(pt,list) or len(pt)!=2 or any(not isinstance(t,(float,int)) or not math.isfinite(t) or t<0 or t>511 for t in pt) for pt in edits):
                raise ValueError('Invalid edited coordinates')
        line_counts[v['status']]+=1
    for k,v in pairs.items():
        if not isinstance(v,dict) or v.get('status') not in PAIR_STATES:
            raise ValueError('Invalid pair decision')
        pair_counts[v['status']]+=1
    return {
        'schema':'s37_s31_local_corrections_audit_aggregate_only_v1',
        'source_photos':2, 'source_crops':39,
        'line_proposals_total':len(known_lines), 'line_proposals_reviewed':len(lines),
        'line_decisions':line_counts,
        'relation_questions_total':len(known_pairs), 'relation_questions_reviewed':len(pairs),
        'relation_decisions':pair_counts,
        'unknown_lines_remain_unknown':len(known_lines)-len(lines),
        'unknown_pairs_remain_unknown':len(known_pairs)-len(pairs),
        'any_physical_ground_truth_confirmed':False,
        'physical_auto_join_authorized':False,
        'ready_to_train_provisional_local_relation_head':sum(pair_counts[k] for k in ('LOCAL_SAME_PROVISIONAL','LOCAL_DIFFERENT_PROVISIONAL'))>=10 and pair_counts['LOCAL_SAME_PROVISIONAL']>0 and pair_counts['LOCAL_DIFFERENT_PROVISIONAL']>0,
        'expert_physics_validation_still_required':True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--private-source-zip',required=True,type=Path)
    ap.add_argument('--corrections',required=True,type=Path)
    ap.add_argument('--aggregate-report',required=True,type=Path)
    a=ap.parse_args()
    with zipfile.ZipFile(a.private_source_zip) as z:
        template=json.loads(z.read('annotations/S31_EMPTY_CORRECTIONS_TEMPLATE.json'))
        drafts=json.loads(z.read('annotations/S31_MACHINE_DRAFTS.json'))
    corrections=json.loads(a.corrections.read_text(encoding='utf-8'))
    result=validate_reviews(template,drafts,corrections)
    a.aggregate_report.parent.mkdir(exist_ok=True,parents=True)
    a.aggregate_report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Provisional line reviews:',result['line_proposals_reviewed'],'/ 133')
    print('Provisional pair reviews:',result['relation_questions_reviewed'],'/ 28')
    print('Real physics ground truth: NOT VERIFIED; automatic physical joining FORBIDDEN')

if __name__=='__main__':main()
