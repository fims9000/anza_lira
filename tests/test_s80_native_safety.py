import numpy as np
import pytest
from scripts.research.bubble_chamber.s80_native_unlabelled_audit import centre_crop


def test_fixed_center_uses_independent_pixels():
    source=np.zeros((512,512),dtype=np.uint8)
    source[160,160]=173
    a=centre_crop(source)
    assert a.shape==(192,192) and a[0,0]==173
    a[0,0]=0
    assert source[160,160]==173
    with pytest.raises(ValueError):centre_crop(np.zeros((192,192),dtype=np.uint8))


def test_source_only_no_physics_label_input():
    from inspect import signature
    from scripts.research.bubble_chamber.s80_native_unlabelled_audit import run
    assert 'native_GT' not in signature(run).parameters
