"""S56 control invariants: data labels synthetic only, no native physics claims."""
import numpy as np
import pytest

from bubble_chamber.s56_global_path_assembly import (
    Edge, make_edge, pair_rows, scored_edges, select_node_mutual,
    select_port_greedy, select_global_forest, forest_components)
from bubble_chamber.s56_three_piece_scenes import TRUE_EDGES,SCENARIOS


def pair(i,j,port_i,port_j,score):
    return Edge(i,j,port_i,port_j,score)


def clean_path_graph():
    return [pair(0,1,-1,0,.98),pair(1,2,-1,0,.93),
            pair(3,4,-1,0,.97),pair(4,5,-1,0,.92),
            pair(0,2,-1,0,.20), pair(1,3,-1,0,.10)]


def test_global_recovers_two_full_three_piece_chains_vs_node_baseline():
    e=clean_path_graph()
    assert len(select_node_mutual(e,threshold=.5,margin=.02))==2
    greedy=select_port_greedy(e,threshold=.5,margin=.02)
    global_sol,qa=select_global_forest(e,threshold=.5,margin=.02)
    assert qa['solver_success'] and qa['native_physics_authorized_joins']==0
    assert {(x.i,x.j) for x in greedy}==TRUE_EDGES
    assert {(x.i,x.j) for x in global_sol}==TRUE_EDGES
    assert forest_components(7,global_sol)==[[0,1,2],[3,4,5],[6]]


def test_global_solution_never_forms_cycle_even_if_all_three_edges_beneficial():
    e=[pair(0,1,-1,0,.98),pair(1,2,-1,0,.96),pair(0,2,0,-1,.90)]
    chosen,qa=select_global_forest(e,threshold=.5,margin=0)
    assert qa['solver_success']
    assert len(chosen)==2
    assert not ({(x.i,x.j) for x in chosen}=={(0,1),(1,2),(0,2)})
    assert len(forest_components(3,chosen))==1


def test_ambiguity_results_in_abstention_not_arbitrary_join():
    e=[pair(0,1,-1,0,.82),pair(0,2,-1,0,.82)]
    assert select_port_greedy(e,threshold=.5,margin=.04)==[]
    chosen,qa=select_global_forest(e,threshold=.5,margin=.04)
    assert chosen==[] and qa['global_ambiguous_abstentions']==1


def test_fail_closed_infinite_cutoff_and_invalid_scores():
    edges=clean_path_graph()
    assert select_global_forest(edges,threshold=float('inf'))[0]==[]
    assert select_node_mutual(edges,threshold=float('inf'))==[]
    assert select_port_greedy(edges,threshold=float('inf'))==[]
    with pytest.raises(ValueError):select_global_forest(edges,threshold=float('nan'))
    with pytest.raises(ValueError):Edge(0,1,0,-1,float('nan'))
    with pytest.raises(ValueError):scored_edges([{'i':0,'j':1,'end_i':0,'end_j':0}],[np.nan])
    with pytest.raises(ValueError):scored_edges([{'i':0,'j':1,'end_i':0,'end_j':0}],[])
    with pytest.raises(ValueError):forest_components(3,[pair(0,1,-1,0,.9),pair(0,2,-1,0,.8)])


def test_polyline_reversal_preserves_geometry_evidence_and_opposite_ports():
    x=np.linspace(50.,72.,31)
    a=np.column_stack((x,80.+.01*(x-60.)**2))
    x2=np.linspace(83.,105.,31)
    b=np.column_stack((x2,81.+.01*(x2-60.)**2))
    c=np.column_stack((x2,110.+.015*(x2-70.)**2))
    base=pair_rows([a,b,c])[0]
    inverse=pair_rows([a[::-1].copy(),b[::-1].copy(),c])[0]
    assert base['i']==inverse['i'] and base['j']==inverse['j']
    assert (base['end_i'],base['end_j'])!= (inverse['end_i'],inverse['end_j'])
    for key in ('endpoint_gap_px','outgoing_tangent_A_to_gap','outgoing_tangent_B_to_gap',
                'outgoing_tangent_mutual','path_length_ratio','chord_curvature_consistency'):
        assert base[key]==pytest.approx(inverse[key],abs=1e-10)


def test_protocol_synthetic_only_and_expected_topology():
    assert len(TRUE_EDGES)==4 and len(SCENARIOS)==5
    assert set(TRUE_EDGES)=={(0,1),(1,2),(3,4),(4,5)}
    with pytest.raises(ValueError):pair_rows([])
