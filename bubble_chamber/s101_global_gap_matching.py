"""S101 topology-safe *global* alternative to greedy S92 gap proposals.

This is a frozen-candidate ABLATION, not a new segmentation, not source pixel
repair, not a physical particle inference or an independently learned model.
Only S92 advisory edges that were geometrically eligible (with third-fragment
and ambiguous-rival abstentions preserved) may enter the optimizer.

Exact maximum-weight linear-forest selection is performed *per connected
component of the S92 component graph* when at most 16 edges: two available
ports per original component, no port repeated, no source-component cycle.
For larger but bounded components the algorithm computes a deterministic
maximum-weight PORT matching and removes cyclic associations (APPROXIMATE,
not a claim of global optimality). Beyond the memory/complexity guard,
ABSTAIN the entire connected group, without silently dropping competitors.

Inputs remain read-only; all returned connections are tentative 2D visual
continuations with unknown physical identity, never observed centerline pixels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import networkx as nx

SCHEMA="S101_BOUNDED_GLOBAL_2D_GAP_PATH_ABLATION_V1"
SELECT="GLOBAL_PATH_REVIEW_NOT_PHYSICAL_ACCEPT"
EXCLUDED="PRESERVED_SOURCE_ABSTENTION"
CONFLICT="GLOBAL_PORT_OR_ACYCLICITY_COMPETITION_ABSTAIN"
BUDGET="GLOBAL_TOPOLOGY_BUDGET_ABSTAIN"

# Original greedy selection, port occupied, degree and cycle decisions are
# RE-OPTIMIZATION candidates; ambiguous partners, neighboring original
# components and unsupported source curves remain ABSTENTIONS.
REOPTIMIZABLE=frozenset((
    "PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT",
    "PORT_OCCUPIED_ABSTAIN",
    "DEGREE_ABSTAIN",
    "CYCLE_ABSTAIN",
))


@dataclass(frozen=True)
class GlobalGapConfig:
    max_exact_edges_per_group:int=16
    max_approx_edges_per_group:int=400
    max_approx_port_nodes:int=320
    max_input_proposals:int=60000

    def validate(self):
        if (any(type(x) is not int for x in vars(self).values())
            or not 1<=self.max_exact_edges_per_group<=20
            or not self.max_exact_edges_per_group<=self.max_approx_edges_per_group<=1500
            or not 2<=self.max_approx_port_nodes<=1024
            or not 1<=self.max_input_proposals<=100000):
            raise ValueError("S101 bounded graph optimization budget invalid")


def _check_proposal(p,known_ids):
    if (not isinstance(p,dict) or
        p.get("status")!="UNKNOWN" or
        p.get("physical_auto_join") is not False or
        p.get("inferred_only") is not True):
        raise ValueError("S101 S92 candidate must be an inferred-only 2D UNKNOWN proposal")
    pid=p.get("proposal_id")
    if not isinstance(pid,str) or not pid or len(pid)>160:
        raise ValueError("S101 S92 candidate identity missing or unbounded")
    a,b=p.get("component_a"),p.get("component_b")
    x,y=p.get("port_a"),p.get("port_b")
    if (not isinstance(a,str) or not isinstance(b,str)
        or a==b or a not in known_ids or b not in known_ids
        or type(x) is not int or type(y) is not int
        or x not in (0,1) or y not in (0,1)):
        raise ValueError("S101 candidate endpoints are not original distinct S86 source ports")
    val=p.get("geometry_score")
    if (not isinstance(val,(int,float)) or isinstance(val,bool)
        or not isfinite(float(val)) or not 0<float(val)<=1.):
        raise ValueError("S101 nonfinite/invalid S92 geometric evidence score")
    if not isinstance(p.get("crosses_other_observed_fragment"),bool):
        raise ValueError("S101 third-fragment obstacle evidence missing")
    decision=p.get("review_decision")
    if not isinstance(decision,str):
        raise ValueError("S101 original S92 machine decision missing")
    # Accept only known machine statuses. Do not allow forged eligibility.
    allowed=REOPTIMIZABLE|frozenset((
        "THIRD_FRAGMENT_NEAR_GAP_ABSTAIN",
        "AMBIGUOUS_PARTNER_ABSTAIN",
        "UNDECIDED_CANDIDATE",
    ))
    if decision not in allowed:
        raise ValueError("S101 S92 review status not recognized")
    if p["crosses_other_observed_fragment"] and decision in REOPTIMIZABLE:
        raise ValueError("S101 prior selected/eligible S92 bridge crosses third source line")
    return (a,x),(b,y),int(round(float(val)*1_000_000))


def _find(parent,x):
    while parent[x]!=x:
        x=parent[x]
    return x


def _exact_group(rows):
    """Exact linear forest with each 2D component endpoint used at most once.

    Graph edges can share an original component if they use different ports.
    Unlike ordinary maximum-weight matching, a cycle of component identities
    is forbidden. Weighted integer objective is source S92 geometry-score sum.
    """
    options=sorted(rows,key=lambda p:(-p["_weight"],p["proposal_id"]))
    n=len(options)
    positive_suffix=[0]*(n+1)
    for i in range(n-1,-1,-1):
        positive_suffix[i]=positive_suffix[i+1]+options[i]["_weight"]
    ids=sorted({str(v) for p in options for v in (
        p["component_a"],p["component_b"]
    )})
    parent0={x:x for x in ids}
    best_weight=-1
    best_tuple=()
    # For small groups (<=16) an exact branch-and-bound search is practical.
    def visit(i,used,parent,chosen,total):
        nonlocal best_weight,best_tuple
        if total+positive_suffix[i]<best_weight:
            return
        if i==n:
            names=tuple(sorted(chosen))
            if total>best_weight or (total==best_weight and names<best_tuple):
                best_weight=total
                best_tuple=names
            return
        p=options[i]
        ua=(p["component_a"],p["port_a"])
        ub=(p["component_b"],p["port_b"])
        a,b=ua[0],ub[0]
        if ua not in used and ub not in used:
            ra=_find(parent,a)
            rb=_find(parent,b)
            if ra!=rb:
                new_parent=parent.copy()
                new_parent[rb]=ra
                visit(i+1,used|{ua,ub},new_parent,
                      chosen+(p["proposal_id"],),total+p["_weight"])
        visit(i+1,used,parent,chosen,total)
    visit(0,set(),parent0,(),0)
    return set(best_tuple)


def _matching_forest_group(rows):
    """Deterministic approximate fallback: port-weight matching then cycle cut.

    This is NOT an exact solution to the maximum-weight linear forest problem.
    Each S92 edge consumes two (component,port) nodes. Cycles are removed
    conservatively in descending S92 score order.
    """
    graph=nx.Graph()
    for p in sorted(rows,key=lambda x:x["proposal_id"]):
        a=(p["component_a"],p["port_a"])
        b=(p["component_b"],p["port_b"])
        # Original S92 candidate port pair should be unique. If the data
        # violates this, the caller validates and fails, not overwrites.
        graph.add_edge(a,b,weight=p["_weight"],proposal_id=p["proposal_id"])
    matched=nx.algorithms.matching.max_weight_matching(
        graph,maxcardinality=False,weight="weight"
    )
    picked={graph.edges[u,v]["proposal_id"] for u,v in matched}
    chosen_rows=sorted(
        (p for p in rows if p["proposal_id"] in picked),
        key=lambda x:(-x["_weight"],x["proposal_id"])
    )
    ids={x for p in chosen_rows for x in (p["component_a"],p["component_b"])}
    parent={x:x for x in ids}
    selected=set()
    for p in chosen_rows:
        a,b=p["component_a"],p["component_b"]
        ra,rb=_find(parent,a),_find(parent,b)
        if ra!=rb:
            parent[rb]=ra
            selected.add(p["proposal_id"])
    return selected


def _check_path_forest(selected):
    used=set()
    parent={}
    for p in selected:
        a,b=p["component_a"],p["component_b"]
        for port in ((a,p["port_a"]),(b,p["port_b"])):
            if port in used:
                raise RuntimeError("S101 violates unique original S86 endpoint constraint")
            used.add(port)
        for v in (a,b):
            parent.setdefault(v,v)
        ra,rb=_find(parent,a),_find(parent,b)
        if ra==rb:
            raise RuntimeError("S101 selected tentative 2D links form a cycle")
        parent[rb]=ra
    return True


def select_global_gap_path_review(
    original_s92_candidates:list[dict],
    *,
    original_source_component_ids:set[str],
    original_bmp_sha256:str,
    config:GlobalGapConfig=GlobalGapConfig(),
):
    config.validate()
    if (not isinstance(original_bmp_sha256,str)
        or len(original_bmp_sha256)!=64):
        raise ValueError("S101 original source BMP SHA256 missing")
    try:int(original_bmp_sha256,16)
    except ValueError as e:raise ValueError("S101 original BMP SHA256 invalid") from e
    if (not isinstance(original_source_component_ids,set)
        or any(not isinstance(x,str) or not x for x in original_source_component_ids)):
        raise ValueError("S101 original source S86 component ID set invalid")
    if (not isinstance(original_s92_candidates,list)
        or len(original_s92_candidates)>config.max_input_proposals):
        raise ValueError("S101 source S92 candidate budget exceeded")
    eligible=[]
    records=[]
    seen=set()
    seen_ports=set()
    original_selected=set()
    for p in original_s92_candidates:
        a,b,w=_check_proposal(p,original_source_component_ids)
        pid=p["proposal_id"]
        if pid in seen:
            raise ValueError("Duplicate S101 source S92 candidate ID")
        seen.add(pid)
        portkey=tuple(sorted((a,b)))
        if portkey in seen_ports:
            raise ValueError("Source S92 contains duplicate endpoint-pair proposals")
        seen_ports.add(portkey)
        if p["review_decision"]=="PRIORITY_GAP_REVIEW_NOT_PHYSICAL_ACCEPT":
            original_selected.add(pid)
        safe=(p["review_decision"] in REOPTIMIZABLE
              and not p["crosses_other_observed_fragment"])
        row={**p,"_weight":w,"_eligible":safe}
        records.append(row)
        if safe:eligible.append(row)
    # Create connected components of SOURCE COMPONENTS, not just ports;
    # different source components cannot be joined into a cycle of tracks.
    rawgraph=nx.Graph()
    for p in eligible:
        rawgraph.add_edge(p["component_a"],p["component_b"])
    groups=sorted(
        (frozenset(group) for group in nx.connected_components(rawgraph)),
        key=lambda group:tuple(sorted(group))
    )
    membership={v:i for i,group in enumerate(groups) for v in group}
    candidates_per_group=[[] for _ in groups]
    for p in eligible:
        candidates_per_group[membership[p["component_a"]]].append(p)
    choice=set()
    info=[]
    blocked=set()
    for gi,grp in enumerate(groups):
        edges=candidates_per_group[gi]
        port_nodes={port for p in edges for port in (
            (p["component_a"],p["port_a"]),
            (p["component_b"],p["port_b"])
        )}
        if len(edges)<=config.max_exact_edges_per_group:
            mode="EXACT_COMPONENT_PORT_LINEAR_FOREST"
            selected=_exact_group(edges)
        elif (len(edges)<=config.max_approx_edges_per_group
              and len(port_nodes)<=config.max_approx_port_nodes):
            mode="APPROX_PORT_MATCHING_THEN_FOREST_FILTER"
            selected=_matching_forest_group(edges)
        else:
            mode="DENSE_GRAPH_BUDGET_ABSTAIN"
            selected=set()
            blocked.update(p["proposal_id"] for p in edges)
        choice.update(selected)
        info.append({
            "source_component_group_index":gi,
            "source_component_count":len(grp),
            "candidate_links":len(edges),
            "original_source_port_nodes":len(port_nodes),
            "solver_mode":mode,
            "alternative_selected_tentative_links":len(selected),
            "exact_global_optimum_claimed_for_this_group":mode.startswith("EXACT_"),
            "physical_particle_identity_verified":False,
        })
    selected=[p for p in records if p["proposal_id"] in choice]
    _check_path_forest(selected)
    output=[]
    for p in records:
        pid=p["proposal_id"]
        if pid in choice:
            decision=SELECT
        elif pid in blocked:
            decision=BUDGET
        elif not p["_eligible"]:
            decision=EXCLUDED
        else:
            decision=CONFLICT
        output.append({
            "proposal_id":pid,
            "component_a":p["component_a"],
            "port_a":p["port_a"],
            "component_b":p["component_b"],
            "port_b":p["port_b"],
            "geometry_score":p["geometry_score"],
            "original_s92_review_decision":p["review_decision"],
            "global_review_decision":decision,
            "globally_selected_tentative_2D_link":decision==SELECT,
            "inferred_only":True,
            "status":"UNKNOWN",
            "physical_auto_join":False,
            "original_centerline_pixels_altered":False,
        })
    old_weight=sum(p["_weight"] for p in records if p["proposal_id"] in original_selected)
    new_weight=sum(p["_weight"] for p in selected)
    cnt=Counter(x["solver_mode"] for x in info)
    return output,{
        "schema":SCHEMA,
        "status":"S101_SAME_S92_CANDIDATE_GLOBAL_LINEAR_FOREST_ALTERNATIVE",
        "source_bmp_sha256":original_bmp_sha256,
        "source_S92_all_candidate_count":len(records),
        "source_S92_reoptimizable_candidate_count":len(eligible),
        "source_S92_original_greedy_selected_count":len(original_selected),
        "S101_global_tentative_selected_count":len(choice),
        "original_s92_greedy_geometry_objective":round(old_weight/1e6,6),
        "s101_global_geometry_objective":round(new_weight/1e6,6),
        "geometry_objective_delta":round((new_weight-old_weight)/1e6,6),
        "original_greedy_retained_pair_ids":sorted(choice&original_selected),
        "newly_selected_proposal_ids":sorted(choice-original_selected),
        "originally_selected_not_kept_ids":sorted(original_selected-choice),
        "preserved_unsafe_or_ambiguous_source_candidates":sum(
            not p["_eligible"] for p in records
        ),
        "bounded_graph_groups":info,
        "solver_mode_counts":dict(cnt),
        "number_of_complexity_abstained_S92_edges":len(blocked),
        "all_selected_original_S86_endpoints_unique":True,
        "no_cycles_in_selected_source_component_graph":True,
        "not_globally_exact_when_matching_fallback_or_budget_abstention":any(
            x["solver_mode"]!="EXACT_COMPONENT_PORT_LINEAR_FOREST" for x in info
        ),
        "all_source_S86_S92_S95_S96_immutable":True,
        "inferred_2D_advisories_not_observed_BMP_pixels":True,
        "no_true_physical_particle_identity_or_trajectory_repaired":True,
        "no_claim_of_validation_improvement_without_independent_S97_S98_reference":True,
        "optimizes_original_S92_heuristic_geometry_scores_only":True,
        "fixed_bounded_solver_configuration":vars(config),
    }
