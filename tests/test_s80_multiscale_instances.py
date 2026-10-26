import numpy as np
import cv2
import pytest
from bubble_chamber.s80_multiscale_instances import augment_pixel_only,choose_cal_only,ARMS,_duplicate


def line(y):
    return np.column_stack((np.linspace(18,145,55),np.full(55,y))).astype(float)


def test_input_contract_and_identity():
    g=np.full((192,192),198,dtype=np.uint8)
    orig=[line(55)]
    x=augment_pixel_only(g,orig,'baseline')
    assert x['polylines'][0] is orig[0]
    assert x['extra_count']==0 and not x['physical_auto_joins_authorized']
    with pytest.raises(TypeError):augment_pixel_only(g,orig,'baseline',truth=[])
    with pytest.raises(ValueError):augment_pixel_only(g.astype(float),orig,'baseline')
    with pytest.raises(ValueError):augment_pixel_only(g,orig,'unknown_arm')


def test_scale_extra_does_not_erase_original():
    g=np.full((192,192),210,dtype=np.uint8)
    cv2.line(g,(18,52),(162,52),17,2)
    cv2.line(g,(18,140),(162,140),20,2)
    orig=[line(52)]
    for arm in ARMS:
        a=augment_pixel_only(g,orig,arm)
        b=augment_pixel_only(g,orig,arm)
        assert a['polylines'][0] is orig[0]
        assert 0<=a['extra_count']<=6
        assert len(a['polylines'])==1+a['extra_count']<=32
        assert a['extra_count']==b['extra_count']
        for p,q in zip(a['polylines'],b['polylines']): np.testing.assert_array_equal(p,q)


def test_duplicates_are_orientation_invariant():
    p=line(60)
    assert _duplicate(p,p[::-1])
    assert not _duplicate(p,line(80))


def test_cal_gate_rejects_extra_false_joins():
    base={'selected_synthetic_correct_links':4,'synthetic_noncorrect_selected':6,
          'complete_three_piece_paths_synthetic':2,'shape_true_fragments_matched':10,
          'solver_failure':0,'added_pixel_instances':0}
    data={arm:{c:dict(base) for c in ('clean','blur_sigma_11')} for arm in ARMS}
    assert choose_cal_only(data)['CAL_selected']=='baseline'
    data['s80_medium_add4']['blur_sigma_11'].update(selected_synthetic_correct_links=7,
        shape_true_fragments_matched=13,synthetic_noncorrect_selected=7)
    assert choose_cal_only(data)['CAL_selected']=='baseline'
    data['s80_medium_add4']['blur_sigma_11']['synthetic_noncorrect_selected']=6
    assert choose_cal_only(data)['CAL_selected']=='s80_medium_add4'
    with pytest.raises(ValueError):choose_cal_only({})
