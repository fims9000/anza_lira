"""Real-coordinate relation study tests use synthetic *coordinate fixture* only."""
import copy
import numpy as np
import pytest
from bubble_chamber.real_sparse_lira import (
    SparseProtocol, _features, _split_track, _accept, make_examples,
    run_sparse_study,
)
from bubble_chamber.visualmetro_import import SCHEMA


def _manifest():
    frames=[]
    for i in range(4):
        tracks=[]
        for track_index in range(3):
            # Multiple close tracks; no artificial perfect segmentation masks.
            xy = [[float(j*17+i), float(80+track_index*20+j*j*0.8)] for j in range(12)]
            tracks.append({'track_id':f'e1t0{track_index+1}', 'n_points':len(xy),
                           'measured_pixel_xy':{str(v):[[x+v*4, y+v*3] for x,y in xy] for v in (1,2,3)}})
        frames.append({'frame_id':f'070_00{i+4}', 'film_id':'070', 'tracks':tracks})
    return {'schema':SCHEMA,'frames':frames}


def test_only_valid_projections_and_truth_not_features():
    config=SparseProtocol(missing_right_fraction=0.0,k_candidates=3)
    m=_manifest()
    rows, audit=make_examples(m,config)
    assert audit['true_possible']==3*3*3*4
    assert audit['candidate_true']==audit['true_possible']
    assert len(rows)==3*3*3*3*4
    assert set(r['view'] for r in rows)=={1,2,3}
    assert all(len(r['geometry_basic'])==3 and len(r['geometry_generic'])==8 and len(r['geometry_axial_fuzzy'])==14 for r in rows)
    assert all(np.isfinite(r['geometry_axial_fuzzy']).all() for r in rows)
    assert sum(r['truth_same_track'] for r in rows)==audit['candidate_true']


def test_cut_hides_real_clicked_point_and_does_not_invent_image():
    xyz=np.asarray([[i*10.,i*3.] for i in range(12)])
    a,b=_split_track(xyz,0.5)
    assert len(a)>=3 and len(b)>=3
    assert not np.array_equal(a[-1],b[0])
    assert 1+len(a)+len(b)<=len(xyz)
    for f in (0.35,0.50,0.65):
        _split_track(np.arange(14).reshape(7,2),f)
    with pytest.raises(ValueError):_split_track(xyz[:5],0.5)


def test_feature_invariance_under_shared_rigid_rotation_translation():
    rng=np.random.default_rng(41)
    a=np.cumsum(rng.normal(size=(7,2))*7,axis=0)
    b=np.cumsum(rng.normal(size=(8,2))*7,axis=0)+a[-1]
    x=_features(a,b)
    theta=1.234
    rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    y=_features(a@rot+75,b@rot+75)
    assert all(np.allclose(a,b,rtol=1e-8,atol=1e-8) for a,b in zip(x,y))


def test_mismatch_and_frame_split():
    with pytest.raises(ValueError,match='validated'):
        make_examples({'schema':'wrong','frames':[]})
    m=_manifest()
    cfg=SparseProtocol(missing_right_fraction=0.,k_candidates=3)
    result, predictions=run_sparse_study(m,cfg)
    assert result['real_images_matched']==0
    assert result['aggregate']['basic']['tp']>=0
    assert len(result['folds'])==4
    for f in result['folds']:
        assert f['test_frame'] not in f['train_frames']
        assert f['validation_frame'] not in f['train_frames']
        assert f['validation_frame']!=f['test_frame']
    assert len(predictions)>0
    assert set(r['test_frame'] for r in predictions)=={f['frame_id'] for f in m['frames']}


def test_global_mutual_best_prevents_conflicting_links():
    rows=[{'scene_id':'one','left_id':'l1','right_id':'r1'},
          {'scene_id':'one','left_id':'l2','right_id':'r1'},
          {'scene_id':'one','left_id':'l1','right_id':'r2'}]
    result=_accept(rows,np.array([.99,.97,.2]),.5,.03)
    assert result.tolist()==[False,False,False] # strong competitor: needs review
    result=_accept(rows,np.array([.99,.91,.2]),.5,.03)
    assert result.tolist()==[True,False,False]
