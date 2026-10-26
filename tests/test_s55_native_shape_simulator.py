"""S55 synthetic geometry adaptation without false native ground truth."""
from __future__ import annotations
from hashlib import sha256
import numpy as np
import pytest
from scripts.research.bubble_chamber.s55_native_shape_simulator import (
    translate_measured_machine_shape,split_unlabeled_source_geometries,
    render_short_native_shapes,PROTOCOL,
)


def curve(length=60.):
    p=np.linspace(0,1,50)
    return np.column_stack((15.+length*p,130.+7*np.sin(p*5)))


def test_translate_only_no_source_geometry_scaling():
    p=curve(70)
    mapped=translate_measured_machine_shape(p,np.random.default_rng(63))
    assert np.all(mapped>=13) and np.all(mapped<=179)
    assert mapped.shape[1]==2
    source_length=float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))
    output_length=float(np.sum(np.linalg.norm(np.diff(mapped,axis=0),axis=1)))
    assert output_length<=source_length+1.e-3
    assert output_length>=15
    assert np.isfinite(mapped).all()


def test_short_stroke_generator_keeps_same_background_and_label_blind():
    raw=np.full((192,192),190,dtype='uint8')
    pts=translate_measured_machine_shape(curve(35),np.random.default_rng(17))
    a=render_short_native_shapes(raw,pts,seed=12,test=True)
    b=render_short_native_shapes(raw,pts,seed=12,test=True)
    assert list(a)==list(b)
    assert len(a)==6
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert all(np.count_nonzero(x<190)>0 for x in a.values())
    assert np.all(raw==190)
    assert len(render_short_native_shapes(raw,pts,seed=12,test=False))==4


def test_source_photo_candidate_id_folds_are_disjoint_and_not_claimed_gt():
    tiles=[]
    for ph,n in [('005',321),('006',246)]:
        per=[]
        for k in range(n):
            per.append({'draft_id':f'c{k}','points_xy':curve().tolist()})
        tiles.append({'photo_id':ph,'crop_id':f'H50Z_030_01_{ph}_not_real',
                      'proposals':per})
    # expand into 39 tiles while retaining exactly 321/246 machine proposal IDs
    source={'native_expert_gt_created':0,'auto_physical_join_allowed':False,
            'tiles':tiles+[dict(photo_id='005',crop_id=f'empty{k}',proposals=[])
                           for k in range(37)]}
    folds=split_unlabeled_source_geometries(source)
    assert len(folds['005']['train'])==256
    assert len(folds['006']['train'])==196
    for ph in ['005','006']:
        a={r['id'] for r in folds[ph]['train']}
        b={r['id'] for r in folds[ph]['cal']}
        assert not a&b
    source['native_expert_gt_created']=1
    with pytest.raises(ValueError):split_unlabeled_source_geometries(source)


def test_protocol_fixed_before_metrics_are_seen():
    assert PROTOCOL['seeds_test']['005']==[20265503,20265504]
    assert PROTOCOL['seeds_train']['006']==20265511
    assert PROTOCOL['native_particle_truth_count']==0
