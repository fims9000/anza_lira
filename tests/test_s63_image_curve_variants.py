import numpy as np
import pytest
from bubble_chamber.s63_image_curve_variants import predict_variants,choose_cal_only,METHODS


def test_method_shapes_provenance_and_no_oracle():
    a=np.full((192,192),200,dtype=np.uint8)
    for x in range(18,174):a[80,x]=15
    p=predict_variants(a)
    assert set(p)==set(METHODS)
    assert all(len(lines)<=32 and all(z.ndim==2 and z.shape[1]==2 for z in lines) for lines in p.values())
    q=predict_variants(a)
    for name in p:
        assert len(p[name])==len(q[name])
        for u,v in zip(p[name],q[name]):np.testing.assert_allclose(u,v)


def test_no_data_leakage_input_api():
    with pytest.raises(ValueError):predict_variants(np.ones((192,192),np.float32))
    with pytest.raises(ValueError):predict_variants(np.ones((32,32),np.uint8))


def test_cal_selector_frozen_constraints_and_ties():
    rows={m:dict(true_links=20,decoy_supported=3,full_paths=2,scenes=48) for m in METHODS}
    assert choose_cal_only(rows)['chosen_on_CAL_only']=='s39_original'
    rows['junction_less']['true_links']=27
    assert choose_cal_only(rows)['chosen_on_CAL_only']=='junction_less'
    rows['junction_less']['decoy_supported']=7
    assert choose_cal_only(rows)['chosen_on_CAL_only']=='s39_original'
    rows['darkness_skeleton']['true_links']=27
    assert choose_cal_only(rows)['chosen_on_CAL_only']=='darkness_skeleton'
    with pytest.raises(ValueError):choose_cal_only({})
