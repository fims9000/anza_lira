import numpy as np
import pytest
from bubble_chamber.large_interpolation import interpolate_gap, evaluate_interpolation

@pytest.mark.parametrize('method', ['linear','quadratic','hermite'])
def test_interpolation_shape_and_collinear_line(method):
    p=np.stack([np.arange(12),3*np.arange(12)-2],axis=1)
    r=interpolate_gap(p[:4],p[8:],4,method)
    assert r.shape==(4,2)
    assert np.allclose(r,p[4:8],atol=1e-6)


def test_hermite_exact_cubic_by_index():
    t=np.arange(15,dtype=float)
    p=np.stack([0.1*t**3+3*t,2*t+4],axis=1)
    r=interpolate_gap(p[:5],p[8:],3,'hermite')
    assert r.shape==(3,2)
    assert np.linalg.norm(r-p[5:8],axis=1).mean()<2.0


def test_invalid_shapes_and_methods():
    with pytest.raises(ValueError):
        interpolate_gap(np.ones((2,2)),np.ones((3,2)),1,'linear')
    with pytest.raises(ValueError):
        interpolate_gap(np.ones((3,2)),np.ones((3,2)),1,'fake')


def test_reject_unaudited_data():
    with pytest.raises(ValueError):
        evaluate_interpolation({'schema':'visualmetro_sparse_measurements_v1'})
