import cv2,numpy as np,pytest
from bubble_chamber.s83_unsaturated_local_support import (
    ARMS,QUANTILES,local_dark_stroke_support,thresholds_from_source_CAL,
    filter_extra_candidates,choose_CAL_only)

def l(y):return np.column_stack((np.linspace(30,161,55),np.full(55,y)))

def test_unsaturated_contrast_increases_with_dark_streak_strength_and_reversal():
    a=np.full((192,192),210,np.uint8)
    mild=a.copy();strong=a.copy()
    cv2.line(mild,(30,80),(161,80),140,2)
    cv2.line(strong,(30,80),(161,80),35,2)
    m=local_dark_stroke_support(mild,l(80));s=local_dark_stroke_support(strong,l(80))
    assert .01<m<s<1,(m,s)
    assert abs(s-local_dark_stroke_support(strong,l(80)[::-1]))<1.e-6
    assert local_dark_stroke_support(a,l(80))<.0001

def test_quantiles_are_not_forced_to_saturation():
    q=thresholds_from_source_CAL([.05,.1,.2,.3,.4])
    assert 0<q['s83_local_q40']<q['s83_local_q70']<1.
    assert thresholds_from_source_CAL([])=={'s83_local_q40':1.,'s83_local_q70':1.}
    with pytest.raises(ValueError):thresholds_from_source_CAL([float('nan')])

def test_keep_source_instances_and_no_source_gt_argument():
    im=np.full((192,192),205,np.uint8);cv2.line(im,(30,105),(161,105),15,2)
    a=l(75);b=l(105);thresholds={'s83_local_q40':.1,'s83_local_q70':.2}
    out=filter_extra_candidates(im,[a],[a,b],'s83_local_q40',thresholds)
    assert out['polylines'][0] is a and out['extra_accepted']==1
    assert not out['physical_auto_joins_authorized']
    with pytest.raises(TypeError):local_dark_stroke_support(im,b,ground_truth='real_particle')


def test_strict_CAL_gate_no_false_proposals_allowed():
    x=dict(selected_synthetic_correct_links=3,synthetic_noncorrect_selected=4,
      complete_three_piece_paths_synthetic=1,shape_true_fragments_matched=20,
      solver_failure=0,extra_accepted=0)
    grid={a:{c:dict(x) for c in ('clean','blur_sigma_11')} for a in ARMS}
    assert choose_CAL_only(grid)['CAL_selected']=='baseline'
    grid['s83_local_q70']['blur_sigma_11'].update(selected_synthetic_correct_links=5,
        shape_true_fragments_matched=23,synthetic_noncorrect_selected=5)
    assert choose_CAL_only(grid)['CAL_selected']=='baseline'
    grid['s83_local_q70']['blur_sigma_11']['synthetic_noncorrect_selected']=4
    assert choose_CAL_only(grid)['CAL_selected']=='s83_local_q70'
