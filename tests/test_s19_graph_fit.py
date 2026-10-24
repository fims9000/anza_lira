from pathlib import Path
import json
import numpy as np
import pytest
from scripts.research.bubble_chamber.s19_fit_reviewed_graph import fit_from_real_pair_file


def write_pair_file(path,rows):
    path.write_text(json.dumps({'schema':'s19_expert_pair_labels_v1','rows':rows,
                                'source_package_sha256':'a'*64,'independent_run_calibrated':False}))


def row(i,label):
    # Source is test fixture simulating human clicks; never authentic review.
    near=i%2==0
    return {'schema':'s19_expert_track_pair_v1','film_id':'H50Z_030','run_id':'H50Z_030_01',
            'frame_id':f'H50Z_030_01_{i//3:03d}.bmp','scene_id':f'roi_{i}',
            'left_segment':0,'right_segment':1,'left_fragment_id':'a','right_fragment_id':'b',
            'source_sha256':'a'*64,
            'left_points_xy':[[1,10+i],[4,10+i],[8,10+i],[11,10+i]],
            'right_points_xy':([[12,10+i],[15,10+i],[18,10+i],[21,10+i]] if near
                               else [[12,45+i],[15,45+i],[18,45+i],[21,45+i]]),
            'expert_label':label,'orientation':'minimum_endpoint_distance_unverified',
            'verified_human':True}


def test_empty_real_review_never_invents_training_pairs(tmp_path):
    p=tmp_path/'pairs.json';write_pair_file(p,[])
    receipt=fit_from_real_pair_file(p,tmp_path/'out')
    assert receipt['human_confirmed_pairs']==0
    assert receipt['status']=='INSUFFICIENT_EXPERT_RELATION_LABELS_ABSTAIN'
    assert receipt['real_automatic_track_merges_allowed'] is False
    assert not (tmp_path/'out/s19_provisional_graph_model.json').exists()


def test_real_review_graph_can_train_provisionally_not_certify(tmp_path):
    p=tmp_path/'pairs.json';write_pair_file(p,[row(i,'same_track' if i%2==0 else 'different_track') for i in range(14)])
    result=fit_from_real_pair_file(p,tmp_path/'out')
    assert result['status']=='TRAINABLE_PROVISIONAL_ONLY'
    assert result['same_track']==7 and result['different_track']==7
    assert not result['real_automatic_track_merges_allowed']
    obj=json.loads((tmp_path/'out/s19_provisional_graph_model.json').read_text())
    assert isinstance(obj,dict) and obj


def test_fake_expert_claim_or_unknown_relation_is_rejected(tmp_path):
    p=tmp_path/'pairs.json';r=row(0,'same_track');r['verified_human']=False
    write_pair_file(p,[r]);
    with pytest.raises(ValueError,match='explicitly reviewed'):
        fit_from_real_pair_file(p,tmp_path/'out')
    r['verified_human']=True;r['expert_label']='uncertain';write_pair_file(p,[r])
    with pytest.raises(ValueError,match='unreviewed pairs'):
        fit_from_real_pair_file(p,tmp_path/'out')
