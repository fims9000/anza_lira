"""S23: use visualMetro-trained geometry only to order HUMAN H50Z reviews.

This NEVER promotes unlabeled real-photo Frangi candidates to true particle
tracks or physical joins. Film-070 manual-click distribution differs sharply
from H50Z one-pixel Frangi centerlines, so score is NOT a probability.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
import re
from pathlib import Path
import numpy as np
from .real_sparse_lira import _features
from .s21_selective_graph import grouped_graph_features

SCHEMA='s23_out_of_domain_review_only_v1'


def observed_pair_rows(package:dict, priority:dict):
    if package.get('schema')!='s19_real_photo_review_package_v1':
        raise ValueError('expected provenance verified source package')
    if priority.get('schema')!='s20_real_photo_review_priority_v1' or priority.get('source_package_sha256')!=package['package_sha256']:
        raise ValueError('priority proposals refer to different real photo package')
    tile_map={t['tile_id']:t for t in package['tiles']}
    rows=[];rejected=[];ids=set()
    for q in priority['pair_questions']:
        tid=q.get('tile_id')
        tile=tile_map.get(tid)
        if not tile or q.get('parent_photo')!=tile['parent_photo']:
            raise ValueError('pair tile or parent photo mismatch')
        candidates={c['id']:c for c in tile['candidates']}
        a,b=q['candidate_a'],q['candidate_b']
        if a not in candidates or b not in candidates or a==b:
            raise ValueError('pair references invalid segment IDs')
        key=(tid,*sorted([a,b]))
        if key in ids:raise ValueError('duplicate real photo pair')
        ids.add(key)
        ea,eb=q.get('source_endpoints',[])
        if ea not in (0,1) or eb not in (0,1):raise ValueError('invalid terminal identity')
        l=np.asarray(candidates[a]['centerline_local_xy_px'],float)
        r=np.asarray(candidates[b]['centerline_local_xy_px'],float)
        if ea==0:l=l[::-1]
        if eb==1:r=r[::-1]
        try:
            basic,generic,axial=_features(l,r)
        except (ValueError,IndexError,np.linalg.LinAlgError):
            rejected.append({'tile_id':tid,'candidate_a':a,'candidate_b':b,'reason':'observed geometry insufficient'})
            continue
        geomscore=basic[0]+2*(basic[1]+basic[2])
        rows.append({'scene_id':tid,'frame_id':tile['source']['parent'],
                     'left_id':a,'right_id':b,'selection_score':float(geomscore),
                     'geometry_axial_fuzzy':axial,'original_s20_affinity':q['score_unvalidated'],
                     'reference_pair_key':key})
    return rows,rejected


def rank_unlabeled_real_pairs(package:dict, priority:dict, model,train_support_cut:float)->dict:
    rows,rejected=observed_pair_rows(package,priority)
    if not rows:raise ValueError('no valid real pairs to offer for human review')
    _,context=grouped_graph_features(rows)
    values=np.asarray(model.predict_proba(context)[:,1],float)
    if not np.isfinite(values).all():raise ValueError('nonfinite transferred model scores')
    pairs=deepcopy(priority['pair_questions'])
    keymap={r['reference_pair_key']:(i,r) for i,r in enumerate(rows)}
    count_ood=0
    for pair in pairs:
        key=(pair['tile_id'],*sorted((pair['candidate_a'],pair['candidate_b'])))
        target=keymap.get(key)
        if target is None:
            pair['s23_noncertified_rank_score']=None
            pair['s23_domain_flag']='INSUFFICIENT_GEOMETRY'
            continue
        i,r=target
        ood=r['geometry_axial_fuzzy'][0]>train_support_cut
        count_ood+=int(ood)
        pair['s23_noncertified_rank_score']=round(float(values[i]),6)
        pair['s23_domain_flag']='OUTSIDE_MEASURED_SCALE_SUPPORT' if ood else 'FILM_AND_ANNOTATION_DOMAIN_UNVERIFIED'
    # Without external calibration, scores can ONLY reorder questions for experts.
    # Unsupported/invalid pairs come last, but NO unapproved pair is removed.
    pairs=sorted(pairs,key=lambda p:(
        p['s23_domain_flag']!='FILM_AND_ANNOTATION_DOMAIN_UNVERIFIED',
        -(p['s23_noncertified_rank_score'] if p['s23_noncertified_rank_score'] is not None else -1),
        p['tile_id'],p['candidate_a'],p['candidate_b']))
    result=deepcopy(priority)
    result['pair_questions']=pairs
    result['s23_protocol']={
       'schema':SCHEMA,'physical_joins_authorized':False,
       'expert_track_labels_created':0,'expert_pair_labels_created':0,
       'trained_on':'visualMetro film 070 expert coords, deliberately omitted clicks',
       'new_images':'H50Z Frangi ROIs only; no real pixel masks',
       'transferred_scores_are_calibrated_probabilities':False,
       'transfer_evaluation_available':False,
       'review_pair_count':len(pairs),'out_of_observed_gap_scale_count':count_ood,
       'insufficient_geometry_count':len(rejected),
       'all_suggested_pairs_remain_human_review_required':True,
       'original_package_sha256':package['package_sha256'],
    }
    result['expert_approved_connections']=0
    result['expert_approved_lines']=0
    return result


def insert_priority_in_workbench(html:str,source_priority:dict,new_priority:dict)->str:
    pat=r'(<script id="s20-priority-payload" type="application/json">)(.*?)(</script>)'
    m=re.search(pat,html,flags=re.S)
    if m is None:raise ValueError('S20 original workbench priority JSON not found')
    embedded=json.loads(m.group(2))
    if embedded!=source_priority:
        raise ValueError('workbench does not embed the expected source priority')
    if new_priority.get('source_package_sha256')!=source_priority.get('source_package_sha256'):
        raise ValueError('workbench photo source digest mismatch')
    copy=json.dumps(new_priority,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    result=html[:m.start(2)]+copy+html[m.end(2):]
    # Explanation is shown in priority panel without changing canonical S19 export.
    marker='Подсказки только по геометрии, не готовые ответы.'
    if marker not in result:raise ValueError('original S20 warning missing')
    result=result.replace(marker,marker+' Очередь пар пересортирована моделью с ДРУГОЙ плёнки: никаких подтверждённых связей, необходима проверка физиком.')
    return result
