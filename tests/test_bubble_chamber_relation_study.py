import numpy as np
from bubble_chamber.relation_study import Protocol, scene, examples, create_split, threshold_at_fpr, run_study


def test_scene_provenance_and_identity_are_deterministic():
    a, b = scene(20261024), scene(20261024)
    assert np.array_equal(a['image'], b['image'])
    assert len(a['segments']) == len(a['identity_side'])
    p = Protocol(train_scenes=6,val_scenes=6,test_scenes=6)
    rows, possible = examples(a,p)
    assert possible >= 1
    assert all('identity_side' not in row for row in rows)


def test_split_group_separation_and_positive_negative_examples():
    p = Protocol(train_scenes=12,val_scenes=12,test_scenes=12)
    splits={name:create_split(p,group=name) for name in ('train','validation','test')}
    sets=[{row['scene_id'] for row in d['rows']} for d in splits.values()]
    assert sets[0].isdisjoint(sets[1])
    assert sets[1].isdisjoint(sets[2])
    assert sets[0].isdisjoint(sets[2])
    for d in splits.values():
        assert d['candidate_true_links'] <= d['total_true_links']
        assert any(row['truth_same_track'] for row in d['rows'])
        assert any(not row['truth_same_track'] for row in d['rows'])


def test_calibration_never_uses_test():
    y=np.array([1,1,0,0,0,0])
    s=np.array([.93,.84,.91,.75,.25,.05])
    thr=threshold_at_fpr(y,s,limit=0.0)
    assert thr>0.91
    assert ((s>=thr)&(y==0)).sum()==0


def test_study_reproducible_small_smoke():
    p=Protocol(train_scenes=12,val_scenes=12,test_scenes=12)
    x=run_study(p)
    assert x['status']=='SYNTHETIC_RELATION_ISOLATION_ONLY'
    assert x['splits']['test']['candidate_true_links']>0
    for model in x['results'].values():
        assert model['test']['automatic_merges_in_real_pilot']==0
        assert 0 <= model['test']['false_positive_rate_candidates'] <= 1
        assert 0 <= model['test']['recall_candidates'] <= 1
