import cv2
import numpy as np
import pytest
from bubble_chamber.s66_image_endpoint_tracing import (
    extend_image_curves,choose_calibration_only,ARMS,POLICY)


def test_original_image_curves_are_unchanged_and_copy_safe():
    im=np.full((192,192),215,np.uint8)
    a=np.column_stack((np.linspace(30,90,21),np.full(21,50)))
    out=extend_image_curves(im,[a],'s39_original')
    assert np.array_equal(out[0],a) and out[0] is not a
    with pytest.raises(ValueError):extend_image_curves(im,[a],'no_GT_cheating')


def test_extension_uses_pixels_and_halts_on_blank():
    im=np.full((192,192),210,np.uint8)
    a=np.column_stack((np.linspace(37,91,21),np.full(21,82)))
    assert np.array_equal(extend_image_curves(im,[a],'ridge_extend_relaxed')[0],a)
    cv2.line(im,(30,82),(102,82),25,2)
    out=extend_image_curves(im,[a],'ridge_extend_relaxed')[0]
    assert len(out)>=len(a)
    assert np.isfinite(out).all()
    assert out[:,0].min()>=7 and out[:,0].max()<185
    assert len(out)<=len(a)+20
    assert set(POLICY)==set(ARMS[1:])


def test_cal_selection_requires_no_extra_false_and_bounded_decoys():
    origin={'synthetic_correct':2,'synthetic_noncorrect':11,
            'matched_source_fragments':31,'synthetic_decoy_matched':7,'scenes':32}
    data={arm:dict(origin) for arm in ARMS}
    assert choose_calibration_only(data)['CAL_only_selected']=='s39_original'
    data['ridge_extend_strict'].update(synthetic_correct=5,synthetic_noncorrect=12)
    assert choose_calibration_only(data)['CAL_only_selected']=='s39_original'
    data['ridge_extend_strict'].update(synthetic_noncorrect=10,synthetic_decoy_matched=7)
    assert choose_calibration_only(data)['CAL_only_selected']=='ridge_extend_strict'
