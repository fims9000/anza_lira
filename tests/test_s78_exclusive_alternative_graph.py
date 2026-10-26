"""S78 source/blind safety and contradictory alternative selectors."""
import numpy as np
import pytest
from bubble_chamber.s56_global_path_assembly import Edge, select_global_forest
from bubble_chamber.s78_exclusive_alternative_graph import (
    Candidate,_consistent,alternatives_from_pixels,select_exclusive_alternatives,
    select_non_destructive,ARMS)


def candidate(i,j,a,b,p,origin='s56_geometry',name=None):
    return Candidate(Edge(i,j,a,b,p),origin,name or f'{origin}-{i}-{j}-{a}-{b}')


def test_missing_hypotheses_bitwise_original_and_non_destructive(monkeypatch):
    import bubble_chamber.s78_exclusive_alternative_graph as s78
    lines=[np.column_stack((np.linspace(20,45,15),np.full(15,70+i*20))) for i in range(3)]
    old=[Edge(0,1,-1,0,.95),Edge(1,2,-1,0,.93)]
    gray=np.full((192,192),200,np.uint8)
    monkeypatch.setattr(s78,'propose_pixel_gap_pairs',lambda *a:[])
    baseline,_=select_global_forest(old,threshold=.8,margin=.04)
    for arm in ARMS:
        out=select_non_destructive(gray,lines,old,.8,arm)
        assert out['polylines'] is lines
        assert out['chosen']==baseline
        assert not out['selected_pixel_alternatives']
        assert out['physical_identity_verified'] is False
        assert out['physics_auto_joins_authorized'] is False


def test_same_port_alternative_is_suppressed_and_every_original_kept(monkeypatch):
    import bubble_chamber.s78_exclusive_alternative_graph as m
    gray=np.full((192,192),200,np.uint8)
    lines=[np.column_stack((np.linspace(20,45,20),np.full(20,70+i*20))) for i in range(3)]
    base=[Edge(0,1,-1,0,.89),Edge(1,2,-1,0,.92)]
    monkeypatch.setattr(m,'propose_pixel_gap_pairs',lambda *a:[(.91,(0,-1),(1,0))])
    p,qa=alternatives_from_pixels(gray,lines,base,'alternative_strict')
    assert len(p)==len(base) and qa['same_port_duplicate_suppressed']==1
    assert qa['candidate_extra_alternatives']==0


def test_competing_ports_same_pair_never_both_selected():
    proposals=[candidate(0,1,-1,0,.91,name='geom'),
               candidate(0,1,0,-1,.99,'s77_pixel_gap','gap'),
               candidate(1,2,0,-1,.95,name='second')]
    got,qa=select_exclusive_alternatives(proposals,cutoff=.80,margin=.04)
    assert qa['solver_success'] and _consistent(got)
    assert sum(c.edge.i==0 and c.edge.j==1 for c in got)<=1
    assert not any(c.id=='geom' for c in got)
    assert 'gap' in [c.id for c in got]


def test_rival_ties_abstain_not_select_two():
    candidates=[candidate(0,1,-1,0,.91,name='a'),
                candidate(0,1,0,-1,.92,'s77_pixel_gap','b')]
    chosen,qa=select_exclusive_alternatives(candidates,cutoff=.80,margin=.04)
    assert chosen==[] and qa['ambiguous_abstentions']==1


def test_pairwise_exclusive_and_global_cycle_cut():
    # A fully overconnected triangle has to return a forest, never a cycle.
    v=[candidate(0,1,-1,0,.96,name='a'),candidate(0,2,0,0,.94,name='b'),
       candidate(1,2,-1,-1,.95,name='c')]
    chosen,qa=select_exclusive_alternatives(v,cutoff=.8,margin=.04)
    assert qa['solver_success'] and _consistent(chosen)
    assert len(chosen)<=2


def test_invalid_inputs_fail_closed(monkeypatch):
    import bubble_chamber.s78_exclusive_alternative_graph as m
    im=np.full((192,192),210,dtype=np.uint8)
    with pytest.raises(ValueError):alternatives_from_pixels(im.astype(float),[],[],ARMS[1])
    with pytest.raises(ValueError):select_exclusive_alternatives([],cutoff=-.1)
    # Failed global optimizer must return no suggested physical connection.
    monkeypatch.setattr(m,'_solve',lambda *a,**k:([],0.,False))
    got,qa=select_exclusive_alternatives([candidate(0,1,-1,0,.95)],cutoff=.8)
    assert not got and not qa['solver_success']
