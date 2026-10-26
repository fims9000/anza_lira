import numpy as np
import pytest
from scripts.research.bubble_chamber.s47_registration_sensitivity import (
    NORMAL_OFFSETS,eval_offset)


def test_offsets_are_fixed_no_surprise_optimizing_on_test():
    assert NORMAL_OFFSETS==(0,2,4,6,8)


def test_sensitivity_responds_to_shift_normal_not_x():
    img=np.full((192,192),200,dtype='uint8')
    path=np.column_stack([np.linspace(30,160,70),np.full(70,80.)])
    same,score1=eval_offset(img,img,path,0)
    displaced,score2=eval_offset(img,img,path,8)
    assert same and not displaced
    assert score1>score2
    with pytest.raises(ValueError):eval_offset(img,img,path,77)
