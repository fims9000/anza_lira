import cv2
import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s84_graph_context_candidates import (
    ARMS,thresholds_from_CAL_pixel_scores,gate_additional_lines,choose_CAL_only)


def curve(x,y):return np.column_stack((np.linspace(x,x+35,31),np.full(31,y)))
def pic():
    im=np.full((192,192),208,np.uint8)
    for y in (70,108):cv2.line(im,(20,y),(55,y),20,2)
    cv2.line(im,(70,70),(105,70),20,2)
    return im


def test_preserve_identity_reject_oracle_parameters_and_nonfinite():
    a,b=curve(20,70),curve(20,108);extra=curve(70,70)
    f=lambda rows:np.full(len(rows),.95)
    thresholds={'q25':0.,'q40':0.}
    out=gate_additional_lines(pic(),[a,b],[a,b,extra],[],f,.70,'baseline',thresholds)
    assert len(out['polylines'])==2 and out['polylines'][0] is a and out['extra_accepted']==0
    out=gate_additional_lines(pic(),[a,b],[a,b,extra],[],f,.70,'s84_photo_q40',thresholds)
    assert out['polylines'][0] is a and out['polylines'][1] is b
    assert out['extra_accepted']==1 and not out['physical_auto_joins_authorized']
    with pytest.raises(TypeError):gate_additional_lines(pic(),[a,b],[a,b,extra],[],f,.7,'baseline',thresholds,source_truth=[1])
    with pytest.raises(ValueError):gate_additional_lines(pic(),[a,b],[a,b.copy(),extra],[],f,.7,'baseline',thresholds)
    with pytest.raises(ValueError):gate_additional_lines(pic(),[a,b],[a,b,extra],[],f,float('nan'),'baseline',thresholds)


def test_graph_weak_or_conflicting_base_edge_cannot_enter():
    a,b=curve(20,70),curve(20,108);extra=curve(70,70)
    thresholds={'q25':0.,'q40':0.}
    low=lambda rows:np.full(len(rows),.20)
    out=gate_additional_lines(pic(),[a,b],[a,b,extra],[],low,.6,'s84_context_single',thresholds)
    assert not out['extra_accepted'] and out['rejected_weak_graph']==1
    # Both ports of baseline's connection are reserved by the source proposal.
    conflicting=[Edge(0,1,-1,0,.95)]
    high=lambda rows:np.full(len(rows),.98)
    out=gate_additional_lines(pic(),[a,b],[a,b,extra],conflicting,high,.6,'s84_context_single',thresholds)
    assert out['extra_accepted']==0


def test_candidate_is_not_forced_to_join_and_reverse_invariance():
    im=pic();a,b=curve(20,70),curve(20,108);extra=curve(70,70)
    f=lambda rows:np.full(len(rows),.91)
    one=gate_additional_lines(im,[a,b],[a,b,extra],[],f,.6,'s84_photo_q40',{'q25':0,'q40':0})
    two=gate_additional_lines(im,[a,b],[a,b,extra[::-1]],[],f,.6,'s84_photo_q40',{'q25':0,'q40':0})
    assert one['extra_accepted']==two['extra_accepted']==1
    assert 'chosen' not in one  # no forced graph predictions


def test_quantiles_and_strict_promotion_gate():
    assert thresholds_from_CAL_pixel_scores([0,.1,.2,.3,.4])==pytest.approx({'q25':.1,'q40':.16})
    assert thresholds_from_CAL_pixel_scores([])=={'q25':1.,'q40':1.}
    with pytest.raises(ValueError):thresholds_from_CAL_pixel_scores([float('nan')])
    base=dict(selected_synthetic_correct_links=3,synthetic_noncorrect_selected=4,
        complete_three_piece_paths_synthetic=1,shape_true_fragments_matched=20,
        solver_failure=0,extra_accepted=0)
    grid={a:{c:dict(base) for c in ('clean','blur_sigma_11')} for a in ARMS}
    assert choose_CAL_only(grid)['CAL_only_selected']=='baseline'
    grid['s84_context_single']['blur_sigma_11'].update(selected_synthetic_correct_links=5,
        shape_true_fragments_matched=22,synthetic_noncorrect_selected=5)
    assert choose_CAL_only(grid)['CAL_only_selected']=='baseline'
    grid['s84_context_single']['blur_sigma_11']['synthetic_noncorrect_selected']=4
    assert choose_CAL_only(grid)['CAL_only_selected']=='s84_context_single'
