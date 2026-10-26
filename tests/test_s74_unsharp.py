import numpy as np
import pytest
from bubble_chamber.s74_unsharp_preprocess import ARMS,preprocess,choose_cal_only


def example():
    arr=np.full((192,192),192,dtype='uint8')
    arr[94:98,22:155]=50
    return arr


def test_pixel_only_deterministic_and_unmodified_original():
    src=example()
    assert ARMS==('identity','unsharp_065','unsharp_130')
    for arm in ARMS:
        a,b=preprocess(src,arm),preprocess(src,arm)
        np.testing.assert_array_equal(a,b)
        assert a.dtype==src.dtype and a.shape==src.shape and not np.shares_memory(a,src)
    np.testing.assert_array_equal(preprocess(src,'identity'),src)
    assert not np.array_equal(preprocess(src,'unsharp_130'),src)
    with pytest.raises(ValueError):preprocess(src.astype(float),'identity')
    with pytest.raises(ValueError):preprocess(src,'unknown')
    with pytest.raises(TypeError):preprocess(src,'identity',ground_truth=[])


def test_selection_requires_blur_gain_without_clean_safety_loss():
    a={k:{'selected_synthetic_correct_links':6,'synthetic_noncorrect_selected':12,
            'complete_three_piece_paths_synthetic':1,'solver_failure':0} for k in ('clean','blur_sigma_11')}
    grid={key:{c:dict(x) for c,x in a.items()} for key in ARMS}
    assert choose_cal_only(grid)['chosen_CAL_only']=='identity'
    grid['unsharp_065']['blur_sigma_11']['selected_synthetic_correct_links']=9
    assert choose_cal_only(grid)['chosen_CAL_only']=='unsharp_065'
    grid['unsharp_065']['clean']['synthetic_noncorrect_selected']=15
    assert choose_cal_only(grid)['chosen_CAL_only']=='identity'
    with pytest.raises(ValueError):choose_cal_only({'identity':grid['identity']})
