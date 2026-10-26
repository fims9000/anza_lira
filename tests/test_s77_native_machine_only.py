import numpy as np
import pytest
from scripts.research.bubble_chamber.s77_native_unlabeled_centre_audit import centre_crop_192


def test_native_centre_crop_is_deterministic_and_label_blind():
    gray=np.arange(512*512,dtype=np.uint32).reshape(512,512).astype('uint8')
    crop=centre_crop_192(gray)
    assert crop.shape==(192,192)
    np.testing.assert_array_equal(crop,gray[160:352,160:352])
    crop[0,0]=0
    assert gray[160,160]!=0
    with pytest.raises(ValueError):centre_crop_192(np.zeros((192,192),dtype=np.uint8))
    with pytest.raises(ValueError):centre_crop_192(gray.astype('float32'))
