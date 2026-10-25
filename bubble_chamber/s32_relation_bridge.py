"""S32: safe bridge from S31 local pair corrections to provisional graph learning.

Local choices are NOT physicist-certified. Unreviewed pairs never become
negative. An empty correction document yields zero supervised relations.
"""
from __future__ import annotations
from collections import Counter
import hashlib,json
import numpy as np
from .s31_machine_first import validate_draft_pack
from .s31_weak_supervision import validate_corrections


def compile_local_relation_reviews(drafts:dict,corrections:dict|None):
    validate_draft_pack(drafts)
    decisions=validate_corrections(drafts,corrections)
    reviewed=corrections.get('relation_decisions',{}) if corrections else {}
    records=[];counts=Counter()
    for photo in drafts['photos']:
        segments={d['draft_id']:d for d in photo['drafts']}
        for rel in photo.get('relation_drafts',[]):
            response=reviewed.get(rel['relation_id'])
            if response is None or response['status']=='LOCAL_UNSURE':continue
            left,right=rel['candidate_a'],rel['candidate_b']
            if left not in segments or right not in segments or left==right:
                raise ValueError('Invalid relation geometry or foreign source image')
            # A locally rejected fragment cannot be used as a positive pair.
            if response['status']=='LOCAL_SAME_PROVISIONAL' and any(
              decisions.get(k,{}).get('status')=='USER_REJECTED_PROVISIONAL' for k in (left,right)):
                raise ValueError('Refuse SAME relation containing rejected line')
            label='same_provisional' if response['status']=='LOCAL_SAME_PROVISIONAL' else 'different_provisional'
            coords=[]
            for did in (left,right):
                d=segments[did]
                edit=decisions.get(did,{}).get('edited_points_xy')
                coords.append(edit if edit is not None else d['points_xy'])
            for path in coords:
                xy=np.asarray(path,float)
                if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():
                    raise ValueError('Cannot emit unverified malformed polyline')
            records.append({'schema':'s32_local_provisional_pair_v1','photo_id':photo['photo_id'],
                'crop_id':photo['crop_id'],'source_gray_pixels_sha256':photo['source_gray_pixels_sha256'],
                'relation_id':rel['relation_id'],'left_id':left,'right_id':right,
                'left_points_xy':coords[0],'right_points_xy':coords[1],
                'local_review_label':label,'source':'LOCAL_USER_PROVISIONAL_NOT_PHYSICIST_CERTIFIED',
                'expert_certified':False,'physics_join_allowed':False})
            counts[label]+=1
    source_digest=hashlib.sha256(json.dumps({'s28':drafts['source_S28_proposals_sha256'],
                             's29':drafts['source_S29_reviewer_sha256']},sort_keys=True).encode()).hexdigest()
    ready=counts['same_provisional']>=5 and counts['different_provisional']>=5
    return {'schema':'s32_relation_review_bridge_v1','source_digest':source_digest,
       'total_candidate_questions':sum(len(p.get('relation_drafts',[])) for p in drafts['photos']),
       'locally_reviewed_same':counts['same_provisional'],
       'locally_reviewed_different':counts['different_provisional'],
       'expert_certified_relation_count':0,
       'eligible_for_provisional_graph_training':ready,
       'eligible_for_physically_automatic_joins':False,
       'reviewed_rows':records,'unknown_candidates_not_used_as_negatives':True}
