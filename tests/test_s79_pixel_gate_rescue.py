import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge
from bubble_chamber.s79_pixel_gate_rescue import ARMS,additional_candidates,solve_s79_from_pixels


def scene():
    im=np.full((192,192),195,np.uint8)
    pieces=[np.column_stack((np.linspace(20,48,12),np.full(12,55+i*26.))) for i in range(3)]
    allrows=[{'i':0,'j':1},{'i':0,'j':2},{'i':1,'j':2}]
    return im,pieces,allrows


def test_no_alternatives_returns_bitwise_original_without_changing_lines(monkeypatch):
    import bubble_chamber.s79_pixel_gate_rescue as m
    im,p,r=scene();base=[Edge(0,1,-1,0,.93)];chosen=base[:]
    monkeypatch.setattr(m,'propose_pixel_gap_pairs',lambda *a:[])
    z=solve_s79_from_pixels(im,p,base,r,[.93,.81,.80],.84,'rescue_relaxed',chosen)
    assert z['polylines'] is p and z['chosen']==chosen
    assert z['selected_candidate_ids']==['S79-G-0-0-1']
    assert z['native_physical_auto_joins_authorized'] is False


def test_pixel_supported_pair_can_escape_strict_gate_only_as_hypothesis(monkeypatch):
    import bubble_chamber.s79_pixel_gate_rescue as m
    im,p,r=scene();old=[Edge(0,1,-1,0,.9)]
    # Pair (1,2) is not in S59 strict pair gate, but exists in S56 all-pairs geometry.
    monkeypatch.setattr(m,'propose_pixel_gap_pairs',lambda *a:[(.86,(2,0),(1,-1))])
    options,qa=additional_candidates(im,p,old,r,[.90,.5,.81],'rescue_strict')
    assert len(options)==2 and qa['rescued_from_strict_pair_gate']==1
    assert qa['original_pair_candidates_retained']==1
    assert options[1].edge.i==1 and options[1].edge.j==2
    assert options[1].edge.end_i==-1 and options[1].edge.end_j==0
    assert options[1].edge.score>.84
    assert options[1].origin=='s77_pixel_gap'


def test_same_pair_same_ports_ignored_no_double_count(monkeypatch):
    import bubble_chamber.s79_pixel_gate_rescue as m
    im,p,r=scene();old=[Edge(0,1,-1,0,.9)]
    monkeypatch.setattr(m,'propose_pixel_gap_pairs',lambda *a:[(.9,(0,-1),(1,0))])
    q,qa=additional_candidates(im,p,old,r,[.9,.7,.8],'rescue_relaxed')
    assert len(q)==1 and qa['duplicates_same_ports_suppressed']==1


def test_no_GT_argument_and_invalid_inputs(monkeypatch):
    import bubble_chamber.s79_pixel_gate_rescue as m
    im,p,r=scene()
    with pytest.raises(ValueError):additional_candidates(im.astype(float),p,[],r,[.9,.8,.7],ARMS[1])
    with pytest.raises(TypeError):additional_candidates(im,p,[],r,[.9,.8,.7],ARMS[1],truth_nodes=[])
    with pytest.raises(ValueError):additional_candidates(im,p,[],r,[.9,.8],ARMS[1])
