import numpy as np
import pytest
from scripts.research.bubble_chamber.s79_native_readonly_review import center,SCHEMA

def test_fixed_center_and_no_particle_truth_in_native_review():
    a=np.arange(512*512,dtype=np.uint8).reshape(512,512)
    q=center(a)
    assert q.shape==(192,192) and np.array_equal(q,a[160:352,160:352])
    q[0,0]=0
    assert a[160,160]!=0
    assert 'UNLABELLED' in SCHEMA

def test_reject_wrong_native_crop_or_dtypes():
    with pytest.raises(ValueError):center(np.zeros((191,191),np.uint8))
    with pytest.raises(ValueError):center(np.zeros((512,512),float))
