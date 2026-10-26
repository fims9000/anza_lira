"""S75 pixel-only local processing and no-test-calibration invariants."""
import cv2
import numpy as np
import pytest
from bubble_chamber.s75_local_ridge_rescue import ARMS, preprocess, choose_cal_only


def test_image_only_api_and_determinism():
    im = np.full((192, 192), 180, np.uint8)
    cv2.line(im, (20, 55), (169, 55), 35, thickness=2)
    for arm in ARMS:
        a = preprocess(im, arm)
        b = preprocess(im, arm)
        assert a.shape == im.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, b)
        assert a.min() >= 0 and a.max() <= 255
        assert not np.shares_memory(a, im)
    np.testing.assert_array_equal(preprocess(im, 'identity'), im)
    with pytest.raises(TypeError):
        preprocess(im, 'identity', generator_nodes=[])
    with pytest.raises(ValueError):
        preprocess(im.astype(np.float32), ARMS[1])
    with pytest.raises(ValueError):
        preprocess(im, 'manual_gt')


def test_blank_photo_unchanged_without_hallucination():
    for value in (0, 70, 128, 255):
        im = np.full((192, 192), value, np.uint8)
        for arm in ARMS:
            np.testing.assert_array_equal(preprocess(im, arm), im)


def test_line_enhancement_is_local_and_never_brightens():
    im = np.full((192, 192), 190, np.uint8)
    cv2.line(im, (30, 100), (162, 100), 110, 2)
    for arm in ARMS[1:]:
        out = preprocess(im, arm)
        assert np.all(out <= im)
        assert np.array_equal(out[:10, :], im[:10, :])


def _metrics(c=8, wrong=12, paths=2, fail=0):
    return {'selected_synthetic_correct_links':c, 'synthetic_noncorrect_selected':wrong,
            'complete_three_piece_paths_synthetic':paths, 'solver_failure':fail}


def test_cal_only_positive_and_fail_closed():
    x={arm:{'clean':_metrics(), 'blur_sigma_11':_metrics(2,9,0)} for arm in ARMS}
    assert choose_cal_only(x)['chosen_CAL_only'] == 'identity'
    x[ARMS[1]]={'clean':_metrics(8,12,2), 'blur_sigma_11':_metrics(4,9,0)}
    assert choose_cal_only(x)['chosen_CAL_only'] == ARMS[1]
    x[ARMS[1]]['blur_sigma_11']['synthetic_noncorrect_selected']=10
    assert choose_cal_only(x)['chosen_CAL_only'] == 'identity'
    x[ARMS[1]]['blur_sigma_11']['synthetic_noncorrect_selected']=9
    x[ARMS[1]]['blur_sigma_11']['solver_failure']=1
    assert choose_cal_only(x)['chosen_CAL_only']=='identity'
    with pytest.raises(ValueError):
        choose_cal_only({'identity':x['identity']})
