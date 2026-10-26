"""S38 source-only tests; no private pixels, track coordinates, or expert labels."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import zipfile

import cv2
import numpy as np
import pytest

from bubble_chamber.s38_review_protocol import (
    review_queues, validate_session, compare_sessions, source_statistics,
)
from scripts.research.bubble_chamber.s38_build_private_workbench import build


def mock_sources():
    photos=[]
    for i in range(39):
        n=4 if i<16 else 3
        drafts=[{'draft_id':f'fake_line_{i}_{j}',
                 'source_method':('LSD','AZ','classical')[j%3],
                 'points_xy':[[1,1],[10,15]],
                 'model_score_UNCALIBRATED':0.51}
                for j in range(n)]
        relations=([{'relation_id':f'fake_pair_{i}','candidate_a':drafts[0]['draft_id'],
                      'candidate_b':drafts[1]['draft_id']}]
                   if i<28 else [])
        photos.append({'crop_id':f'fake_crop_{i}','photo_id':'005' if i<21 else '006',
                       'source_gray_pixels_sha256':'','drafts':drafts,'relation_drafts':relations})
    evidence={'rows':[{'relation_id':f'fake_pair_{i}', 'candidate_a':f'fake_line_{i}_0',
                       'candidate_b':f'fake_line_{i}_1', 'crop_id':f'fake_crop_{i}',
                       'photo_id':'005' if i<21 else '006',
                       'measured_endpoint_gap_px':4+i, 'provisional_scores_NOT_PHYSICAL_PROBABILITIES':{}}
                      for i in range(28)]}
    assert sum(len(p['drafts']) for p in photos)==133
    template={'schema':'s31_machine_corrections_v1',
              'machine_source_s28_sha256':'sha_source_A',
              'machine_source_s29_sha256':'sha_source_B',
              'all_unreviewed_stay_unknown':True,'physical_joins_authorized':False,
              'crops':[{'crop_id':p['crop_id']} for p in photos],
              'decisions':{},'relation_decisions':{}}
    return template,{'photos':photos},evidence


def make_session(template, reviewer, round_name):
    result=deepcopy(template)
    result['s38_review_session']={'protocol':'s38_independent_blind_v1',
        'reviewer_id':reviewer,'round':round_name,'scores_revealed':False,
        'physical_gt_certified':False}
    return result


def test_queues_cover_every_item_and_balance_source_strata_without_labels():
    t,d,e=mock_sources()
    r=review_queues(d,e)
    for round_name in ('A','B'):
        assert len(r[round_name]['lines'])==len(set(r[round_name]['lines']))==133
        assert len(r[round_name]['pairs'])==len(set(r[round_name]['pairs']))==28
        assert {'fake_line_0_0','fake_line_38_2'}<=set(r[round_name]['lines'])
        assert {'fake_pair_0','fake_pair_27'}<=set(r[round_name]['pairs'])
        first=r[round_name]['lines'][:10]
        assert len(set(x.split('_')[2] for x in first))>=4
    assert r['A']['lines']!=r['B']['lines']
    assert r==review_queues(d,e)
    inv=source_statistics(d,e,r)
    assert inv['machine_lines']==133 and inv['machine_pairs']==28
    assert inv['expert_certified_real_track_masks']==0
    assert not inv['native_pixel_accuracy_measurable']


def test_no_provisional_pair_without_two_visible_reviewed_lines():
    t,d,e=mock_sources()
    s=make_session(t,'operator_A','A')
    s['relation_decisions']['fake_pair_0']={'status':'LOCAL_SAME_PROVISIONAL'}
    with pytest.raises(ValueError,match='requires both lines'):validate_session(t,d,e,s)
    for k in ('fake_line_0_0','fake_line_0_1'):
        s['decisions'][k]={'status':'USER_ACCEPTED_PROVISIONAL'}
    audit=validate_session(t,d,e,s)
    assert audit['relation_questions_reviewed']==1
    assert audit['blinded_review'] and not audit['physical_auto_join_authorized']
    s['decisions']['fake_line_0_1']={'status':'USER_REJECTED_PROVISIONAL'}
    with pytest.raises(ValueError,match='requires both lines'):validate_session(t,d,e,s)


def test_malicious_sessions_and_degenerate_curves_are_blocked():
    t,d,e=mock_sources()
    s=make_session(t,'operator_A','A')
    for field, value in [('reviewer_id','bad user name'),('round','C'),
                         ('scores_revealed',None),('physical_gt_certified',True)]:
        x=deepcopy(s);x['s38_review_session'][field]=value
        with pytest.raises(ValueError):validate_session(t,d,e,x)
    for pts in ([[1,1],[1,1]], [[1,1],[512,1]], [[1,1],[float('nan'),1]],
                [[0,0]]*129):
        x=deepcopy(s)
        x['decisions']['fake_line_0_0']={'status':'USER_ACCEPTED_PROVISIONAL','edited_points_xy':pts}
        with pytest.raises(ValueError):validate_session(t,d,e,x)
    x=deepcopy(s)
    x['decisions']['fake_line_0_0']={'status':'USER_REJECTED_PROVISIONAL',
                                   'edited_points_xy':[[1.,1.],[5.,6.]]}
    with pytest.raises(ValueError):validate_session(t,d,e,x)


def test_two_independent_rounds_never_promote_to_physical_truth():
    t,d,e=mock_sources()
    a,b=make_session(t,'physicist_A','A'),make_session(t,'physicist_B','B')
    for s in (a,b):
        for j in (0,1):
            s['decisions'][f'fake_line_0_{j}']={'status':'USER_ACCEPTED_PROVISIONAL'}
        s['relation_decisions']['fake_pair_0']={'status':'LOCAL_SAME_PROVISIONAL'}
    a['decisions']['fake_line_1_0']={'status':'USER_ACCEPTED_PROVISIONAL',
                                 'edited_points_xy':[[0.,0.],[10.,15.]]}
    b['decisions']['fake_line_1_0']={'status':'USER_ACCEPTED_PROVISIONAL',
                                 'edited_points_xy':[[0.,0.],[11.,16.]]}
    private,public=compare_sessions(t,d,e,a,b)
    assert public['pair_provisional_agreements']==1
    assert public['line_provisional_agreements']==2
    assert public['line_edit_disagreements']==1
    assert not public['physical_identity_gt_verified']
    assert not public['automatic_particle_track_joins']
    assert not public['physics_reconstruction_authorized']
    assert len(private['tasks'])==158  # 161 total - 3 agreements
    assert 'physicist_A' not in json.dumps(public)
    with pytest.raises(ValueError,match='different'):
        bad=deepcopy(b);bad['s38_review_session']['reviewer_id']='physicist_A'
        compare_sessions(t,d,e,a,bad)
    with pytest.raises(ValueError,match='rounds'):
        bad=deepcopy(b);bad['s38_review_session']['round']='A'
        compare_sessions(t,d,e,a,bad)


def test_round_conflicts_and_uncertain_labels_not_training_target():
    t,d,e=mock_sources();a,b=make_session(t,'audit_A','A'),make_session(t,'audit_B','B')
    a['decisions']['fake_line_2_0']={'status':'USER_ACCEPTED_PROVISIONAL'}
    b['decisions']['fake_line_2_0']={'status':'USER_REJECTED_PROVISIONAL'}
    a['decisions']['fake_line_3_0']={'status':'USER_UNCERTAIN'}
    private,public=compare_sessions(t,d,e,a,b)
    assert public['line_label_disagreements']==1
    assert public['line_incomplete_or_unsure']==1
    assert public['pair_provisional_agreements']==0
    assert not public['ready_for_physics_adjudication']
    assert len(private['tasks'])==161


def test_private_builder_verifies_39_original_pixel_sha_and_blocks_drift(tmp_path):
    from hashlib import sha256
    template,d,e=mock_sources()
    pix=np.arange(512*512,dtype=np.uint32).reshape(512,512)%256
    gray=pix.astype(np.uint8)
    ok,raw=cv2.imencode('.png',gray)
    assert ok
    digest=sha256(gray.tobytes()).hexdigest()
    for p in d['photos']:
        p['source_gray_pixels_sha256']=digest
    archive=tmp_path/'synth_private_source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('annotations/S31_MACHINE_DRAFTS.json',json.dumps(d))
        z.writestr('annotations/S31_EMPTY_CORRECTIONS_TEMPLATE.json',json.dumps(template))
        z.writestr('review/S35_28_REAL_H50Z_PAIR_QUESTIONS_PRIVATE.json',json.dumps(e))
        for p in d['photos']:
            z.writestr('source_39_H50Z_crops/'+p['crop_id']+'.png',raw.tobytes())
    package=tmp_path/'synth_private_reviewer.zip'
    pub=tmp_path/'synth_public_aggregate.json'
    res=build(archive,package,pub)
    assert res['machine_lines']==133
    with zipfile.ZipFile(package) as z:
        assert len([p for p in z.namelist() if '/crops/' in p])==39
        html=z.read('S38_REVIEWER/index.html').decode('utf8')
        assert '<script' in html and 'round_a' not in html
        assert '__PRIVATE_JSON__' not in html
        assert 'physically' not in json.dumps(res)
    assert 'fake_line_' not in pub.read_text()
    tampered=deepcopy(d)
    tampered['photos'][0]['source_gray_pixels_sha256']='bad'
    changed=tmp_path/'source_bad.zip'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(changed,'w') as out:
        for item in src.namelist():
            content=src.read(item)
            if item=='annotations/S31_MACHINE_DRAFTS.json': content=json.dumps(tampered).encode('utf8')
            out.writestr(item,content)
    with pytest.raises(ValueError,match='checksum mismatch'):
        build(changed,tmp_path/'must_not_use.zip')


def test_all_crops_including_zero_proposal_strata_have_independent_queue():
    t,d,e=mock_sources()
    # simulate missing machine proposals while preserving total counts by
    # moving two proposals from one crop to another (source-only synthetic fixture)
    first=d['photos'][38]
    moved=first['drafts']
    first['drafts']=[]
    d['photos'][22]['drafts'].extend(moved)
    q=review_queues(d,e)
    assert len(q['A']['crops'])==39 and len(q['B']['crops'])==39
    assert first['crop_id'] in q['A']['crops']
    assert first['crop_id'] not in q['A']['lines']
    public=source_statistics(d,e,q)
    assert public['crops_without_machine_line_proposals']==1


def test_manually_added_traces_and_crop_review_are_provisional_only():
    t,d,e=mock_sources()
    a,b=make_session(t,'reviewer_a','A'),make_session(t,'reviewer_b','B')
    a['s38_manual_lines']=[{'manual_id':'manual_0001',
                            'crop_id':'fake_crop_38',
                            'points_xy':[[3.,5.],[12.,18.]],
                            'status':'USER_DRAWN_PROVISIONAL'}]
    a['s38_crop_assessments']={'fake_crop_38':{'status':'REVIEWED_FOR_MISSED_TRACKS_LOCAL'}}
    b['s38_crop_assessments']={'fake_crop_38':{'status':'NO_ADDITIONAL_LOCAL'}}
    checked=validate_session(t,d,e,a)
    assert checked['manual_provisional_new_lines']==1
    assert checked['crop_visual_reviews']==1
    private,public=compare_sessions(t,d,e,a,b)
    assert public['crops_reviewed_in_both_rounds']==1
    assert public['manual_new_line_proposals_round_a']==1
    assert public['manual_new_line_proposals_round_b']==0
    assert public['manual_new_line_proposals_auto_matched']==0
    assert any(v['kind']=='new_manual_line' for v in private['tasks'])
    assert any(v['kind']=='crop' for v in private['tasks'])
    assert not public['physical_identity_gt_verified']

    bad=deepcopy(a)
    bad['s38_crop_assessments']['fake_crop_38']['status']='NO_ADDITIONAL_LOCAL'
    with pytest.raises(ValueError,match='contradicts'):validate_session(t,d,e,bad)
    for edits in ([[1,1],[1,1]], [[0,0],[700,10]], [[0,0],[True,1]], [[0,0],[float('inf'),0]]):
        bad=deepcopy(a)
        bad['s38_manual_lines'][0]['points_xy']=edits
        with pytest.raises(ValueError):validate_session(t,d,e,bad)
    bad=deepcopy(a)
    bad['s38_manual_lines'].append(deepcopy(bad['s38_manual_lines'][0]))
    with pytest.raises(ValueError,match='Duplicate'):validate_session(t,d,e,bad)
    bad=deepcopy(a)
    bad['s38_manual_lines'][0]['crop_id']='nonexistent_crop'
    with pytest.raises(ValueError,match='unknown source crop'):validate_session(t,d,e,bad)


def test_inter_reviewer_kappa_is_conditional_and_never_accuracy():
    t,d,e=mock_sources()
    a,b=make_session(t,'human_A','A'),make_session(t,'human_B','B')
    ids=['fake_line_0_0','fake_line_0_1','fake_line_1_0','fake_line_1_1']
    for i,did in enumerate(ids):
        a['decisions'][did]={'status':'USER_ACCEPTED_PROVISIONAL' if i<2 else 'USER_REJECTED_PROVISIONAL'}
        b['decisions'][did]={'status':'USER_ACCEPTED_PROVISIONAL' if i in (0,1,2) else 'USER_REJECTED_PROVISIONAL'}
    _,r=compare_sessions(t,d,e,a,b)
    x=r['line_two_label_agreement']
    assert x['co_reviewed_two_label_n']==4
    assert x['observed_agreement']==.75
    assert x['cohens_kappa']==.5
    assert r['pair_two_label_agreement']['cohens_kappa'] is None
    assert r['agreement_is_not_accuracy']
    assert r['human_certified_native_track_masks']==0


def test_provenance_promotions_and_boolean_coordinate_injection_blocked():
    t,d,e=mock_sources()
    a=make_session(t,'tester_a','A')
    a['decisions']['fake_line_0_0']={'status':'USER_ACCEPTED_PROVISIONAL',
        'review_origin':'PHYSICIST_CERTIFIED'}
    with pytest.raises(ValueError,match='provenance'):validate_session(t,d,e,a)
    a['decisions']['fake_line_0_0']['review_origin']='LOCAL_USER_PROVISIONAL_NOT_PHYSICIST_CERTIFIED'
    a['decisions']['fake_line_0_0']['edited_points_xy']=[[True,1],[30,40]]
    with pytest.raises(ValueError,match='coordinates'):validate_session(t,d,e,a)
