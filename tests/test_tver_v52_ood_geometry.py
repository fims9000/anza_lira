"""OOD geometry renderer and metric smoke tests, no geology or training."""
import numpy as np
import pytest
from scripts.tver_v52_ood_geometry import ood_scene, metrics, REGIMES

@pytest.mark.parametrize('family',REGIMES)
def test_ood_scene_observations_are_finite_and_label_blind(family):
    sample=ood_scene(5000,family)
    a,b=sample['proposals'][0]['source'],sample['proposals'][0]['target']
    assert np.isfinite(sample['gray']).all()
    assert sample['gray'].shape==(96,96)
    y0,y1,x0,x1=sample['rectangle']
    assert np.count_nonzero(sample['visible_prob'][y0:y1,x0:x1])==0
    assert a!=b
    assert sample['hidden_centerline']
    assert sample['kind']=='LINK'

@pytest.mark.parametrize('family',REGIMES)
def test_matching_truth_is_success(family):
    sample=ood_scene(5002,family)
    m=metrics(sample,sample['hidden_centerline'])
    assert m['successful']
    assert abs(m['symmetric_mean_distance_px'])<1e-6

def test_no_route_counts_failure():
    sample=ood_scene(5001,REGIMES[0])
    assert not metrics(sample,None)['successful']
    assert metrics(sample,None)['symmetric_mean_distance_px'] is None