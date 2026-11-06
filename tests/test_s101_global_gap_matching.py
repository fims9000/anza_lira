"""S101 greedy vs global *same S92 candidate pool* exact topology ablation.

Synthetic graphs here are model geometric hypotheses, never real H50Z
physicist labels. Tests insist on immutable originals and abstentions.
"""
from __future__ import annotations

from copy import deepcopy
import pytest

from bubble_chamber.s101_global_gap_matching import (
    SCHEMA,SELECT,EXCLUDED,BUDGET,GlobalGapConfig,
    select_global_gap_path_review,_check_path_forest
)


def edge(pid,a,pa,b,pb,score,decision="PORT_OCCUPIED_ABSTAIN",
         crosses=False):
    return {
        "proposal_id":pid,"component_a":a,"port_a":pa,
        "component_b":b,"port_b":pb,"geometry_score":score,
        "review_decision":decision,"crosses_other_observed_fragment":crosses,
        "status":"UNKNOWN","physical_auto_join":False,"inferred_only":True,
        "inferred_bridge_points_xy":[[0.,0.],[1.,1.]],
    }


def test_greedy_one_high_bridge_prevents_two_higher_total_alternatives():
    # Greedy chooses A-B (0.9), occupying A.p1 and B.p0, thereby blocking
    # A-C (0.8) and B-D (0.8). Global chooses two compatible 0.8 edges.
    rows=[
        edge("AB","A",1,"B",0,.9,
             decision="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        edge("AC","A",1,"C",0,.8),
        edge("BD","B",0,"D",1,.8),
    ]
    saved=deepcopy(rows)
    r,m=select_global_gap_path_review(
        rows,original_source_component_ids={"A","B","C","D"},
        original_bmp_sha256="a"*64
    )
    assert rows==saved
    assert {x["proposal_id"] for x in r if x["globally_selected_tentative_2D_link"]}=={"AC","BD"}
    assert m["source_S92_original_greedy_selected_count"]==1
    assert m["S101_global_tentative_selected_count"]==2
    assert m["geometry_objective_delta"]==pytest.approx(.7)
    assert m["solver_mode_counts"]=={"EXACT_COMPONENT_PORT_LINEAR_FOREST":1}
    assert all(x["status"]=="UNKNOWN" and x["physical_auto_join"] is False
               and x["original_centerline_pixels_altered"] is False for x in r)


def test_rival_and_near_third_line_s92_abstentions_are_immutable():
    rows=[
        edge("AB","A",1,"B",0,.7,
             decision="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT"),
        edge("BC","B",1,"C",0,.9,decision="AMBIGUOUS_PARTNER_ABSTAIN"),
        edge("CD","C",1,"D",0,.8,decision="THIRD_FRAGMENT_NEAR_GAP_ABSTAIN",
             crosses=True),
    ]
    r,m=select_global_gap_path_review(
        rows,original_source_component_ids={"A","B","C","D"},
        original_bmp_sha256="b"*64
    )
    assert [x["proposal_id"] for x in r if x["globally_selected_tentative_2D_link"]]==["AB"]
    assert {x["proposal_id"] for x in r if x["global_review_decision"]==EXCLUDED}=={"BC","CD"}
    assert m["preserved_unsafe_or_ambiguous_source_candidates"]==2


def test_endpoint_uniqueness_and_cycle_in_component_graph_not_just_matching():
    # Every component uses different endpoint ports but a triangular cycle
    # is a fake closed particle continuation in this restricted S101 model.
    rows=[
        edge("AB","A",1,"B",0,.9),
        edge("BC","B",1,"C",0,.85),
        edge("CA","C",1,"A",0,.81),
    ]
    results,meta=select_global_gap_path_review(
        rows,original_source_component_ids={"A","B","C"},
        original_bmp_sha256="c"*64
    )
    chosen={x["proposal_id"] for x in results if x["globally_selected_tentative_2D_link"]}
    assert chosen=={"AB","BC"}
    assert meta["no_cycles_in_selected_source_component_graph"]
    assert len(chosen)==2


def test_deterministic_equal_weight_ties_across_source_input_permutations():
    rows=[
        edge("AB","A",1,"B",0,.8),
        edge("AC","A",1,"C",0,.8),
        edge("BD","B",0,"D",1,.8),
    ]
    all_runs=[]
    for order in (rows,rows[::-1],[rows[1],rows[0],rows[2]]):
        result,meta=select_global_gap_path_review(
            deepcopy(order),original_source_component_ids={"A","B","C","D"},
            original_bmp_sha256="d"*64
        )
        all_runs.append((
            {x["proposal_id"] for x in result if x["globally_selected_tentative_2D_link"]},
            meta["s101_global_geometry_objective"]
        ))
    assert all(x==all_runs[0] for x in all_runs)
    assert all_runs[0][0]=={"AC","BD"}


def test_approximate_matching_explicitly_not_the_global_optimum():
    # Force approximate solver using a single connected 4-edge source group.
    rows=[
        edge("AB","A",1,"B",0,.90),
        edge("AC","A",1,"C",0,.80),
        edge("BD","B",0,"D",1,.80),
        edge("CD","C",1,"D",0,.60),
    ]
    output,meta=select_global_gap_path_review(
        rows,original_source_component_ids={"A","B","C","D"},
        original_bmp_sha256="e"*64,
        config=GlobalGapConfig(max_exact_edges_per_group=2,
                               max_approx_edges_per_group=6,
                               max_approx_port_nodes=10)
    )
    assert meta["solver_mode_counts"]=={"APPROX_PORT_MATCHING_THEN_FOREST_FILTER":1}
    assert meta["not_globally_exact_when_matching_fallback_or_budget_abstention"]
    assert meta["no_cycles_in_selected_source_component_graph"]
    assert all(x["physical_auto_join"] is False for x in output)


def test_bounded_component_abstention_never_cherry_picks_dense_group():
    rows=[
        edge("AB","A",1,"B",0,.90),
        edge("BC","B",1,"C",0,.85),
        edge("CD","C",1,"D",0,.78),
        edge("DE","D",1,"E",0,.72),
        edge("EF","E",1,"F",0,.67),
    ]
    result,meta=select_global_gap_path_review(
        rows,original_source_component_ids={"A","B","C","D","E","F"},
        original_bmp_sha256="f"*64,
        config=GlobalGapConfig(max_exact_edges_per_group=2,
                               max_approx_edges_per_group=4,
                               max_approx_port_nodes=20)
    )
    assert len(result)==5
    assert all(x["global_review_decision"]==BUDGET for x in result)
    assert meta["number_of_complexity_abstained_S92_edges"]==5
    assert meta["S101_global_tentative_selected_count"]==0
    assert meta["not_globally_exact_when_matching_fallback_or_budget_abstention"]


def test_duplicate_source_ports_proposal_ids_invalid_score_and_physical_claim_rejected():
    source={"A","B","C"}
    bads=[
        [edge("dup","A",1,"B",0,.8),edge("dup","B",1,"C",0,.6)],
        [edge("ab","A",1,"B",0,.8),edge("ba","B",0,"A",1,.7)],
        [edge("non","A",1,"X",0,.8)],
        [edge("phys","A",1,"B",0,.8)|{"physical_auto_join":True}],
        [edge("bad","A",1,"B",0,float("nan"))],
        [edge("ob","A",1,"B",0,.8)|{"inferred_only":False}],
        [edge("bad","A",1,"B",0,.8,
             decision="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",crosses=True)]
    ]
    for source_rows in bads:
        with pytest.raises(ValueError):
            select_global_gap_path_review(
                source_rows,original_source_component_ids=source,
                original_bmp_sha256="a"*64
            )


def test_empty_and_zero_reoptimizable_keep_undefined_improvement_not_fake_success():
    empty,meta=select_global_gap_path_review(
        [],original_source_component_ids={"A"},
        original_bmp_sha256="a"*64
    )
    assert empty==[]
    assert meta["S101_global_tentative_selected_count"]==0
    unsafe,metadata=select_global_gap_path_review(
        [edge("AB","A",1,"B",0,.8,decision="AMBIGUOUS_PARTNER_ABSTAIN")],
        original_source_component_ids={"A","B"},
        original_bmp_sha256="a"*64
    )
    assert unsafe[0]["global_review_decision"]==EXCLUDED
    assert metadata["S101_global_tentative_selected_count"]==0
    assert metadata["preserved_unsafe_or_ambiguous_source_candidates"]==1



def test_exact_branch_and_bound_matches_independent_bruteforce_for_seeded_graphs():
    from itertools import combinations
    import random

    rng=random.Random(20261106)
    universe=("A","B","C","D","E")
    for case in range(16):
        edges=[]
        used_pairs=set()
        while len(edges)<7:
            a,b=sorted(rng.sample(universe,2))
            pa,pb=rng.randrange(2),rng.randrange(2)
            k=(a,pa,b,pb)
            if k in used_pairs:continue
            used_pairs.add(k)
            edges.append(edge(
                f"seed{case}_edge{len(edges)}",a,pa,b,pb,
                round(rng.uniform(.2,.99),4)
            ))
        # Independent direct enumeration of every 2^7 subset: do not call
        # S101's solver or union-find logic to construct expected objective.
        best=0.
        for n in range(len(edges)+1):
            for idxs in combinations(range(len(edges)),n):
                chosen=[edges[i] for i in idxs]
                ports=[
                    (e["component_a"],e["port_a"]) for e in chosen
                ]+[
                    (e["component_b"],e["port_b"]) for e in chosen
                ]
                if len(ports)!=len(set(ports)):
                    continue
                # Independent graph forest test using Euler's relation:
                # for an undirected simple graph, a forest has |E| = |V|-cc.
                graph={}
                for e in chosen:
                    a,b=e["component_a"],e["component_b"]
                    graph.setdefault(a,set()).add(b)
                    graph.setdefault(b,set()).add(a)
                seen=set()
                cc=0
                for node in graph:
                    if node in seen:continue
                    cc+=1
                    todo=[node]
                    while todo:
                        v=todo.pop()
                        if v in seen:continue
                        seen.add(v)
                        todo.extend(graph[v]-seen)
                if len(chosen)!=len(graph)-cc:
                    continue
                best=max(best,sum(x["geometry_score"] for x in chosen))
        _,meta=select_global_gap_path_review(
            edges,original_source_component_ids=set(universe),
            original_bmp_sha256="a"*64
        )
        assert meta["s101_global_geometry_objective"]==pytest.approx(best,abs=1e-6)
        assert meta["solver_mode_counts"]=={"EXACT_COMPONENT_PORT_LINEAR_FOREST":1}
