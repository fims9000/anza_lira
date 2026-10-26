"""Source-only S37 assertions against leakage, untrusted corrections and source drift."""
from copy import deepcopy
import numpy as np
import pytest
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames,ARMS
from scripts.research.bubble_chamber.s37_validate_local_corrections import validate_reviews
from bubble_chamber.s37_source_model import verify_model_source


def fake_source():
    photos=[]
    for i in range(39):
        n=4 if i<16 else 3
        photos.append({'crop_id':f'fake_{i}',
                       'drafts':[{'draft_id':f'd_{i}_{j}'} for j in range(n)],
                       'relation_drafts':[{'relation_id':f'r_{i}_{j}'} for j in range(1 if i<28 else 0)]})
    # exactly 133 drafts (16*4 + 23*3 = 133)
    assert sum(len(p['drafts']) for p in photos)==133
    return {'photos':photos}


def template():
    return {'schema':'s31_machine_corrections_v1','machine_source_s28_sha256':'alpha',
            'machine_source_s29_sha256':'beta','all_unreviewed_stay_unknown':True,
            'physical_joins_authorized':False,'crops':[{'crop_id':'fake'}],
            'decisions':{},'relation_decisions':{}}


def test_frame_split_145_every_frame_disjoint():
    m={'frames':[{'frame_id':f'070_{i:03d}'} for i in range(145)]}
    tr,ca,te,digest=split_source_frames(m)
    assert (len(tr),len(ca),len(te))==(79,30,36)
    assert not tr&ca and not tr&te and not ca&te
    assert tr|ca|te=={v['frame_id'] for v in m['frames']}
    assert '070_309' not in tr|ca|te # fake ids range 000..144
    assert split_source_frames(m)==(tr,ca,te,digest)
    with pytest.raises(ValueError):split_source_frames({'frames':m['frames'][:-1]})
    with pytest.raises(ValueError):split_source_frames({'frames':m['frames'][:-1]+[m['frames'][0]]})


def test_source_locked_original_models_untouched():
    assert len(verify_model_source())==2
    assert len(ARMS)==7 and 'geometry+AZUNet' in ARMS and 'geometry+UNet' in ARMS


def test_invalid_and_malicious_corrections_rejected():
    orig=template();base=fake_source()
    r=validate_reviews(orig,base,deepcopy(orig))
    assert r['unknown_lines_remain_unknown']==133 and r['unknown_pairs_remain_unknown']==28
    assert r['any_physical_ground_truth_confirmed'] is False
    for mutation in [lambda t:t.update(physical_joins_authorized=True),
                     lambda t:t.update(crops=[]),
                     lambda t:t['decisions'].update({'made_up':{'status':'USER_ACCEPTED_PROVISIONAL'}}),
                     lambda t:t['relation_decisions'].update({'made_up':{'status':'LOCAL_SAME_PROVISIONAL'}}),
                     lambda t:t.update(all_unreviewed_stay_unknown=False)]:
        t=deepcopy(orig);mutation(t)
        with pytest.raises(ValueError):validate_reviews(orig,base,t)


def test_correct_statuses_stay_provisional_and_pair_gate():
    orig=template();s=fake_source();t=deepcopy(orig)
    for i in range(12):
        t['relation_decisions'][f'r_{i}_0']={'status':'LOCAL_SAME_PROVISIONAL' if i%2 else 'LOCAL_DIFFERENT_PROVISIONAL'}
    t['decisions']['d_0_0']={'status':'USER_ACCEPTED_PROVISIONAL','edited_points_xy':[[1.,2.],[3.,4.]]}
    result=validate_reviews(orig,s,t)
    assert result['ready_to_train_provisional_local_relation_head']
    assert result['unknown_pairs_remain_unknown']==16
    assert not result['physical_auto_join_authorized']
    t['decisions']['d_0_0']['edited_points_xy']=[[1.,float('nan')],[2.,3.]]
    with pytest.raises(ValueError):validate_reviews(orig,s,t)
