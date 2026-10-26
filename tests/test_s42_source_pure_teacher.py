"""Hard anti-leakage and no-fictitious-native-truth tests for S42 teacher."""
import numpy as np
import cv2
import pytest
from bubble_chamber.s42_source_pure_teacher import geometry_only_pseudo


def test_teacher_only_geometry_and_unknown_preservation():
    gray=np.full((512,512),210,dtype=np.uint8)
    cv2.line(gray,(30,40),(400,410),30,2)
    x=geometry_only_pseudo(gray,[],(0,0),'005')
    y=geometry_only_pseudo(gray,[],(0,0),'005')
    assert x['AZ_model_used_in_teacher'] is False
    assert x['S31_machine_draft_used_in_teacher'] is False
    assert x['native_expert_gt_used'] is False
    assert np.array_equal(x['mask'],y['mask'])
    assert set(np.unique(x['mask'])).issubset({0,1,255})
    assert (x['mask']==255).sum()>0


def test_known_fiducial_never_used_as_assumed_positive_or_negative():
    gray=np.full((512,512),210,dtype=np.uint8)
    cv2.line(gray,(30,250),(490,250),20,3)
    fid=[{'photo':'005','label':'f1','center_xy_PRIVATE':[250,250],
          'machine_ambiguity_radius_px':45.,'arms_angles_rad_PRIVATE':[]}]
    raw=geometry_only_pseudo(gray,[],(0,0),'005')
    safe=geometry_only_pseudo(gray,fid,(0,0),'005')
    assert safe['mask'][250,250]==255
    assert int(np.count_nonzero(safe['mask']==1))<=int(np.count_nonzero(raw['mask']==1))
    assert int(np.count_nonzero(safe['mask']==0))<=int(np.count_nonzero(raw['mask']==0))


def test_source_pure_teacher_rejects_wrong_image_size():
    with pytest.raises(ValueError):geometry_only_pseudo(np.ones((64,64),dtype=np.uint8),[],(0,0),'005')
