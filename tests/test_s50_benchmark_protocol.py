"""S50 no-leakage and scene-level controls, independent of private photos."""
from __future__ import annotations
import numpy as np
import pytest
from scripts.research.bubble_chamber.s50_benchmark_curvature_continuation import (
    ARMS, PROTOCOL, flatten, fit_model, shuffle_extra_features,
    scene_paired_bootstrap_auc_difference, _public_decision_stats)
from bubble_chamber.s50_curvature_continuation import S50_FIELDS, ALL_FIELDS


def fake(n=24):
    scenes=[]
    for scene in range(n):
        rows=[]
        for i in range(2):
            for j in range(3):
                r={k:(float(i==j)+.01*scene+.001*j) for k in ALL_FIELDS}
                r.update(left_index=i,right_index=j,label_SYNTHETIC_ONLY=int(i==j))
                rows.append(r)
        scenes.append({'pairs':rows,'scenario':'crossing'})
    return scenes


def test_frozen_protocol_does_not_touch_test_for_fit():
    assert PROTOCOL['test_seeds']['005']==(20265003,20265004)
    assert PROTOCOL['test_seeds']['006']==(20265013,20265014)
    assert ARMS['geometry+curvature'][:6]==ARMS['geometry']
    s=fake()
    train,labels=flatten(s)
    model=fit_model(train,labels,ARMS['geometry'])
    before=model(train)
    for row in s[0]['pairs']:row['label_SYNTHETIC_ONLY']=1-row['label_SYNTHETIC_ONLY']
    after=model(train)
    assert np.allclose(before,after)


def test_control_shuffle_never_modifies_source_geometry_labels():
    source, y = flatten(fake())
    old=[dict(q) for q in source]
    shuffled=shuffle_extra_features(source,np.random.default_rng(77))
    assert source==old
    assert np.allclose([r['endpoint_gap_px'] for r in source],
                       [r['endpoint_gap_px'] for r in shuffled])
    assert np.array_equal(y,[r['label_SYNTHETIC_ONLY'] for r in shuffled])
    assert len(shuffled)==len(source)
    with pytest.raises(ValueError):shuffle_extra_features(source[:-1],np.random.default_rng(77))


def test_synthetic_ci_is_scene_clustered_and_public_has_no_per_scene_payload():
    cases=fake(18); rows,y=flatten(cases)
    s=np.array([.9 if v else .1 for v in y]); t=s.copy()
    result=scene_paired_bootstrap_auc_difference(cases,s,t,seed=17,n_bootstrap=50)
    assert result['delta_AUROC_curvature_minus_geometry']==0
    assert result['conditional_synthetic_scene_bootstrap_95pct_percentile']==[0.,0.]
    assert result['independent_real_physical_run_confidence_interval'] is False
    summary=_public_decision_stats(cases,s,.8)
    assert summary['synthetic_wrong_joins']==0 and summary['native_physics_automatic_links']==0
    assert 'per_scene_counts_private' not in summary


def test_invalid_fit_and_unmatched_scene_shapes_fail_closed():
    cases=fake(20);r,y=flatten(cases)
    with pytest.raises(ValueError):scene_paired_bootstrap_auc_difference(cases,np.zeros(10),np.zeros(10),seed=1)
    with pytest.raises(ValueError):flatten([{'pairs':r[:5]}])
