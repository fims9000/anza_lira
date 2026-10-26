"""S50 hard negative and source restrictions without private photos."""
from __future__ import annotations
import numpy as np
import pytest
from bubble_chamber.s50_curvature_continuation import (
    S50_FIELDS, ALL_FIELDS, _oriented_end_fit,
    curvature_continuation_features, pair_geometry_and_curvature,
    matched_candidates)


def arc(start: float, stop: float, *, k: float = .004, y: float = 96., n: int = 31):
    x = np.linspace(start, stop, n)
    return np.stack((x, y + k * (x-90.)**2), axis=-1)


def test_finite_local_curvature_true_vs_different_sign():
    a = arc(20, 81)
    good = arc(103, 164)
    wrong = arc(103, 164, k=-.01)
    z = curvature_continuation_features(a, good)
    w = curvature_continuation_features(a, wrong)
    assert set(z) == set(S50_FIELDS)
    assert all(np.isfinite(list(z.values())))
    assert z[S50_FIELDS[0]] < w[S50_FIELDS[0]]
    assert set(pair_geometry_and_curvature(a,good)) == set(ALL_FIELDS)


def test_reversing_either_fragment_preserves_curvature_and_geometry():
    a = arc(20, 81, k=.002)
    b = arc(103,164,k=.002)
    orig = pair_geometry_and_curvature(a,b)
    for aa,bb in [(a[::-1],b),(a,b[::-1]),(a[::-1],b[::-1])]:
        result = pair_geometry_and_curvature(aa,bb)
        assert np.allclose(list(result.values()),list(orig.values()),atol=1e-7)


def test_fixed_length_nondestructive_and_validate():
    x = arc(10, 90, n=51)
    y = arc(110, 182, n=59)
    z = arc(112, 172, n=31)
    left,right = matched_candidates([x,x+2],[y,y+2,z])
    assert [len(q) for q in left+right] == [31]*5
    assert len(x)==51 and len(y)==59
    with pytest.raises(ValueError):matched_candidates([x], [y,y,z])
    with pytest.raises(ValueError):matched_candidates([x,x], [y,y,z[:10]])


def test_fail_closed_nonfinite_and_degenerate():
    a=arc(20,81)
    b=arc(103,164)
    with pytest.raises(ValueError):curvature_continuation_features(a,np.full((31,2),np.nan))
    with pytest.raises(ValueError):_oriented_end_fit(np.zeros((31,2)), -1)
    with pytest.raises(ValueError):_oriented_end_fit(np.zeros((5,2)), -1)
    with pytest.raises(ValueError):curvature_continuation_features(a,b*0 + a[-1])


def test_native_truth_remains_unknown():
    import inspect
    import bubble_chamber.s50_curvature_continuation as s50
    source=inspect.getsource(s50)
    assert 'neither a calibrated' in source.lower()
    assert 'label_SYNTHETIC_ONLY' not in source
