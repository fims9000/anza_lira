import numpy as np
import pytest
from ct28_endpoint_jitter import jitter_batch


class FixedNoise:
    def random(self):
        return 0.0

    def normal(self, loc, scale, size):
        return np.array([[2., 0., 0.], [0., 0., 0.]])[:size[0]]


def sample():
    g = np.indices((12, 12, 12), dtype=np.float32)
    x = np.zeros((1, 4, 12, 12, 12), dtype=np.float32)
    x[0, 0] = g[0]/12
    y = np.zeros((1, 1, 12, 12, 12), dtype=np.float32)
    y[0, 0, 5, 5, 5] = 1
    for channel, point in enumerate(([4., 5., 5.], [7., 5., 5.]), 1):
        x[0, channel] = np.exp(-np.sum((g-np.array(point)[:,None,None,None])**2,axis=0)/4.5)
    return x, y


def test_recenter_and_independent_endpoint_motion():
    x, y = sample()
    original_x, original_y = x.copy(), y.copy()
    a, b, n = jitter_batch(x, y, degree=2, voxel_mm=1, endpoint_sigma_mm=1.5, noise_mm=.8, rng=FixedNoise())
    assert n == 1
    assert np.unravel_index(a[0,1].argmax(), a.shape[2:]) == (5,5,5)
    assert np.unravel_index(a[0,2].argmax(), a.shape[2:]) == (6,5,5)
    assert b[0,0,4,5,5] == 1
    assert a[0,0,4,5,5] == pytest.approx(x[0,0,5,5,5])
    assert not a[0,3].any()
    assert np.array_equal(x, original_x) and np.array_equal(y, original_y)


def test_seeded_and_binary():
    x,y = sample()
    kw = dict(degree=2, voxel_mm=1, endpoint_sigma_mm=1.5, noise_mm=.8)
    a,b,n = jitter_batch(x,y,rng=np.random.default_rng(2),**kw)
    c,d,m = jitter_batch(x,y,rng=np.random.default_rng(2),**kw)
    assert n==m and np.array_equal(a,c) and np.array_equal(b,d)
    assert set(np.unique(b)) <= {0,1}


def test_shape_validation():
    x,y=sample()
    with pytest.raises(ValueError):
        jitter_batch(x[:,:2],y,degree=2,voxel_mm=1,endpoint_sigma_mm=1.5,noise_mm=.8,rng=FixedNoise())
