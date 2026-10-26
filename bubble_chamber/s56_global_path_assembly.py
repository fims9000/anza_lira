"""S56 machine-only geometric path *suggestions*, not physical trajectories.

The 2-D forest assumption is a restricted open-curve synthetic hypothesis.
It is NOT a universal physical constraint: real spirals and junctions can exist.
All selected edges are reversible proposals with unknown native identity.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .s35_fragment_link import optimal_endpoints, pair_features, GEOMETRY_FIELDS


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    end_i: int
    end_j: int
    score: float

    def __post_init__(self):
        if (self.i < 0 or self.j <= self.i or self.end_i not in (0, -1)
                or self.end_j not in (0, -1) or not np.isfinite(self.score)):
            raise ValueError('Invalid finite scored and oriented candidate edge')

    def ports(self):
        return (self.i, self.end_i), (self.j, self.end_j)


def make_edge(i: int, j: int, a: Any, b: Any, score: float) -> Edge:
    if i >= j:
        raise ValueError('Require i<j')
    pick = optimal_endpoints(a, b)
    return Edge(i, j, int(pick['endpoint_index_a']), int(pick['endpoint_index_b']), float(score))


def pair_rows(polylines: list[np.ndarray]) -> list[dict[str, Any]]:
    if not 3 <= len(polylines) <= 32:
        raise ValueError('Graph size outside bounded supported range')
    out=[]
    for i,j in combinations(range(len(polylines)), 2):
        features=pair_features(polylines[i],polylines[j],evidence={})
        chosen=optimal_endpoints(polylines[i],polylines[j])
        out.append({'i': i, 'j': j,
                    'end_i': int(chosen['endpoint_index_a']),
                    'end_j': int(chosen['endpoint_index_b']),
                    **{name:float(features[name]) for name in GEOMETRY_FIELDS}})
    return out


def scored_edges(rows: list[dict], scores: Any) -> list[Edge]:
    s=np.asarray(scores,dtype=float)
    if s.shape != (len(rows),) or not np.isfinite(s).all():
        raise ValueError('Score count and finiteness required')
    edges=[Edge(int(row['i']),int(row['j']),int(row['end_i']),int(row['end_j']),float(v))
           for row,v in zip(rows,s)]
    if len({(e.i,e.j) for e in edges})!=len(edges):
        raise ValueError('Duplicate undirected pair')
    return edges


def _rivals(edges: list[Edge], *, by_port: bool) -> dict[Any,list[Edge]]:
    group=defaultdict(list)
    for e in edges:
        a,b=e.ports()
        group[a if by_port else e.i].append(e)
        group[b if by_port else e.j].append(e)
    return group


def _margin_eligible(edges: list[Edge], threshold: float, margin: float, *, by_port: bool) -> list[Edge]:
    if np.isnan(threshold) or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid cutoff/margin')
    groups=_rivals(edges,by_port=by_port)
    allowed=[]
    for e in edges:
        if not e.score>threshold:
            continue
        a,b=e.ports()
        rivals_a=[r.score for r in groups[a if by_port else e.i] if r != e]
        rivals_b=[r.score for r in groups[b if by_port else e.j] if r != e]
        if rivals_a and e.score-max(rivals_a)<margin-1e-12:continue
        if rivals_b and e.score-max(rivals_b)<margin-1e-12:continue
        allowed.append(e)
    return allowed


def _forest_and_ports_valid(selection: list[Edge]) -> bool:
    if len({port for e in selection for port in e.ports()})!=2*len(selection):
        return False
    if not selection:return True
    g=nx.Graph()
    g.add_edges_from((e.i,e.j) for e in selection)
    return nx.is_forest(g)


def select_node_mutual(edges: list[Edge], *, threshold:float, margin:float=.04) -> list[Edge]:
    """S39-style strong node-level abstain; one edge per fragment."""
    eligible=_margin_eligible(edges,threshold,margin,by_port=False)
    result=[];used=set()
    for e in sorted(eligible,key=lambda e:(-e.score,e.i,e.j)):
        if e.i in used or e.j in used:continue
        used.update((e.i,e.j));result.append(e)
    return result


def select_port_greedy(edges: list[Edge], *, threshold:float, margin:float=.04) -> list[Edge]:
    """One join per 2-D endpoint; no automatic physical join."""
    eligible=_margin_eligible(edges,threshold,margin,by_port=True)
    result=[]
    for e in sorted(eligible,key=lambda e:(-e.score,e.i,e.j)):
        if _forest_and_ports_valid(result+[e]):result.append(e)
    return result


def _global_milp(eligible: list[Edge], *, threshold:float, banned:frozenset[int]=frozenset(),
                 time_limit:float=3.) -> tuple[list[int],float,bool]:
    """Exact maximum weight degree-constrained *forest* via iterative cycle cuts.

    On solver failure or time limit: return empty, score=0, success=False.
    """
    if not eligible or not np.isfinite(threshold):
        return [],0.,True
    m=len(eligible)
    ports=defaultdict(list)
    for k,e in enumerate(eligible):
        for p in e.ports():ports[p].append(k)
    cycles=[]
    n_cuts=0
    weights=np.asarray([max(0.,e.score-threshold) for e in eligible],float)
    while n_cuts<=m:
        cons=lil_matrix((len(ports)+len(cycles),m),dtype=float)
        upper=[]
        for p,idx in ports.items():
            row=len(upper)
            for k in idx:cons[row,k]=1
            upper.append(1.)
        for row_indices in cycles:
            row=len(upper)
            for k in row_indices:cons[row,k]=1
            upper.append(float(len(row_indices)-1))
        low=np.zeros(m,float)
        high=np.ones(m,float)
        for k in banned:high[k]=0.
        res=milp(-weights,integrality=np.ones(m,dtype=int),bounds=Bounds(low,high),
                 constraints=LinearConstraint(cons.tocsr(),
                    np.full(len(upper),-np.inf),np.asarray(upper)),
                 options={'time_limit':time_limit, 'mip_rel_gap':0., 'disp':False})
        if res.status!=0 or res.x is None:
            return [],0.,False
        selected=[k for k,v in enumerate(res.x) if v>.5]
        graph=nx.Graph()
        for k in selected:
            e=eligible[k];graph.add_edge(e.i,e.j,k=k)
        detected=nx.cycle_basis(graph)
        if not detected:
            return selected,float(sum(weights[k] for k in selected)),True
        for cycle in detected:
            members=[]
            for a,b in zip(cycle,cycle[1:]+cycle[:1]):
                members.append(graph[a][b]['k'])
            cycles.append(tuple(members))
            n_cuts+=1
    return [],0.,False


def select_global_forest(edges: list[Edge], *, threshold:float, margin:float=.04,
                         timeout:float=3.) -> tuple[list[Edge],dict]:
    """All proposed joins remain unverified. Fails CLOSED on MILP failure."""
    if np.isnan(threshold) or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid policy')
    eligible=[e for e in edges if e.score>threshold]
    if not eligible or not np.isfinite(threshold):
        return [],{'solver_success':True,'global_ambiguous_abstentions':0,
                   'native_physics_authorized_joins':0}
    ids,objective,ok=_global_milp(eligible,threshold=threshold,time_limit=timeout)
    if not ok:
        return [],{'solver_success':False,'global_ambiguous_abstentions':len(eligible),
                   'native_physics_authorized_joins':0}
    stable=[];uncertain=0
    for k in ids:
        _,counterfactual,ok2=_global_milp(eligible,threshold=threshold,
                    banned=frozenset((k,)),time_limit=timeout)
        if not ok2:
            return [],{'solver_success':False,'global_ambiguous_abstentions':len(eligible),
                       'native_physics_authorized_joins':0}
        if objective-counterfactual>=margin-1e-10:
            stable.append(eligible[k])
        else:
            uncertain+=1
    if not _forest_and_ports_valid(stable):
        raise AssertionError('MILP returned inconsistent selection')
    return stable,{'solver_success':True,'global_ambiguous_abstentions':uncertain,
                   'native_physics_authorized_joins':0}


def forest_components(n_nodes:int, chosen:list[Edge]) -> list[list[int]]:
    if not 0<=n_nodes<=32 or any(e.j>=n_nodes for e in chosen):
        raise ValueError('Invalid node count')
    if not _forest_and_ports_valid(chosen):
        raise ValueError('Invalid returned path forest')
    g=nx.Graph();g.add_nodes_from(range(n_nodes))
    g.add_edges_from((e.i,e.j) for e in chosen)
    return [sorted(x) for x in sorted(nx.connected_components(g),key=lambda x:min(x))]
