"""S78: non-destructive alternate machine-endpoint graph with exclusivity.

No synthetic reference or physical particle ID enters this module. Its input is
image-derived line geometry, previously fitted frozen S56 pair scores, and S77
pixel-supported gap hypotheses. All original line instances remain intact.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .s56_global_path_assembly import Edge, select_global_forest
from .s77_short_gap_instances import propose_pixel_gap_pairs

ARMS=('baseline','alternative_strict','alternative_relaxed')
SOURCE_S77={'alternative_strict':'short_gap_strict','alternative_relaxed':'short_gap_relaxed'}
MAX_EXTRA=32
MARGIN=.04


@dataclass(frozen=True)
class Candidate:
    edge: Edge
    origin: str  # s56_geometry or s77_pixel_gap
    id: str

    def __post_init__(self):
        if self.origin not in ('s56_geometry','s77_pixel_gap') or not self.id:
            raise ValueError('Invalid candidate provenance')


def alternatives_from_pixels(gray: np.ndarray, polylines: list[np.ndarray],
                             baseline: list[Edge], arm: str) -> tuple[list[Candidate],dict]:
    """All original edges retained; alternatives never mutate or replace lines.

    S77 raw local proposals are admitted only for a S59-strict S56 pair, but
    may represent different endpoint ports. Identical ports mean the SAME
    hypothesis, not an extra independent piece of evidence.
    """
    if arm not in ARMS or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Frozen pixel-only arm and uint8 image required')
    if len(polylines)>32:
        raise ValueError('Too many original machine curves')
    b={ (e.i,e.j):e for e in baseline }
    if len(b)!=len(baseline):raise ValueError('Original pair duplication')
    result=[Candidate(e,'s56_geometry',f'S78-G-{i}-{e.i}-{e.j}') for i,e in enumerate(baseline)]
    stats={'raw_local_gap_proposals':0,'same_port_duplicate_suppressed':0,
           'outside_strict_pair_gate':0,'candidate_extra_alternatives':0,
           'original_graph_candidates_retained':len(baseline),
           'original_polylines_retained':len(polylines)}
    if arm=='baseline' or len(polylines)<2:return result,stats
    proposals=propose_pixel_gap_pairs(gray,polylines,SOURCE_S77[arm])
    stats['raw_local_gap_proposals']=len(proposals)
    for raw_score,p,q in proposals:
        (i,pi),(j,pj)=p,q
        if i==j:continue
        if j<i:i,j,pi,pj=j,i,pj,pi
        original=b.get((i,j))
        if original is None:
            stats['outside_strict_pair_gate']+=1;continue
        if (original.end_i,original.end_j)==(pi,pj):
            stats['same_port_duplicate_suppressed']+=1;continue
        if not 0<=raw_score<=1 or not np.isfinite(raw_score):
            raise ValueError('Unbounded pixel evidence score')
        # Ranking score only. It is deliberately NOT called a physical or
        # calibrated probability; both terms are fixed before S78 CAL.
        score=min(.999,max(original.score+.06,.68+.24*float(raw_score)))
        edge=Edge(i,j,pi,pj,score)
        result.append(Candidate(edge,'s77_pixel_gap',f'S78-P-{arm}-{i}-{j}-{pi}-{pj}'))
        stats['candidate_extra_alternatives']+=1
        if stats['candidate_extra_alternatives']>MAX_EXTRA:
            raise ValueError('Frozen extra gap hypothesis budget exceeded')
    if len({c.id for c in result})!=len(result):raise ValueError('Duplicate public-safe candidate ID')
    return result,stats


def _consistent(chosen: list[Candidate]) -> bool:
    port=set();pair=set();g=nx.Graph()
    for x in chosen:
        e=x.edge
        if any(p in port for p in e.ports()) or (e.i,e.j) in pair:
            return False
        port.update(e.ports());pair.add((e.i,e.j));g.add_edge(e.i,e.j)
    return len(g)==0 or nx.is_forest(g)


def _solve(eligible: list[Candidate], cutoff: float, *, banned=frozenset(), timeout=3.) -> tuple[list[int],float,bool]:
    """Cycle-cutting MILP, supporting same-pair *competing* edge variants."""
    if not eligible:return [],0.,True
    if len(eligible)>528:raise ValueError('Unexpected candidate count')
    groups=defaultdict(list);pair_groups=defaultdict(list)
    for k,c in enumerate(eligible):
        e=c.edge
        for p in e.ports():groups[p].append(k)
        pair_groups[(e.i,e.j)].append(k)
    constraints=[(indexes,1.) for indexes in groups.values()]
    constraints += [(indexes,1.) for indexes in pair_groups.values() if len(indexes)>1]
    m=len(eligible)
    weights=np.array([c.edge.score-cutoff for c in eligible],float)
    if np.any(weights<=0):raise ValueError('Only above-cutoff hypotheses eligible')
    if not (np.isfinite(weights).all() and np.isfinite(timeout) and timeout>0):
        raise ValueError('Unsafe optimization inputs')
    bounds_upper=np.ones(m,float)
    for k in banned:
        if k<0 or k>=m:raise ValueError('Invalid banned candidate')
        bounds_upper[k]=0.
    for _ in range(m+1):
        A=lil_matrix((len(constraints),m),dtype=float)
        upper=[]
        for r,(members,cap) in enumerate(constraints):
            for k in members:A[r,k]=1
            upper.append(float(cap))
        try:
            opt=milp(-weights,integrality=np.ones(m,dtype=int),
                bounds=Bounds(np.zeros(m),bounds_upper),
                constraints=LinearConstraint(A.tocsr(),np.full(len(upper),-np.inf),np.asarray(upper)),
                options={'time_limit':timeout,'mip_rel_gap':0.,'disp':False})
        except Exception:
            return [],0.,False  # solver errors fail closed, never greedy fallback
        if opt.status!=0 or opt.x is None:return [],0.,False
        selected=[k for k,v in enumerate(opt.x) if v>.5]
        graph=nx.Graph()
        for k in selected:
            e=eligible[k].edge
            if graph.has_edge(e.i,e.j):
                raise AssertionError('Duplicate pair escaped pair constraints')
            graph.add_edge(e.i,e.j,candidate_index=k)
        cycles=nx.cycle_basis(graph)
        if not cycles:
            if not _consistent([eligible[k] for k in selected]):
                raise AssertionError('Inconsistent forest/ports returned')
            return selected,float(weights[selected].sum()),True
        for cyc in cycles:
            members=[graph[u][v]['candidate_index'] for u,v in zip(cyc,cyc[1:]+cyc[:1])]
            constraints.append((members,float(len(members)-1)))
    return [],0.,False


def select_exclusive_alternatives(candidates: list[Candidate], *,
                                  cutoff: float, margin: float=MARGIN,
                                  timeout: float=3.) -> tuple[list[Candidate],dict]:
    if not np.isfinite(cutoff) or not 0<=cutoff<=1 or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid frozen threshold/margin')
    if len({x.id for x in candidates})!=len(candidates):raise ValueError('Unstable candidate IDs')
    eligible=[x for x in candidates if x.edge.score>cutoff]
    if not eligible:
        return [],dict(solver_success=True,ambiguous_abstentions=0,
                       selected_pixel_alternatives=0,physical_auto_joins_authorized=False)
    ids,objective,ok=_solve(eligible,cutoff,timeout=timeout)
    if not ok:
        return [],dict(solver_success=False,ambiguous_abstentions=len(eligible),
                       selected_pixel_alternatives=0,physical_auto_joins_authorized=False)
    chosen=[];abstained=0
    for i in ids:
        _,counterfactual,ok2=_solve(eligible,cutoff,banned=frozenset({i}),timeout=timeout)
        if not ok2:
            return [],dict(solver_success=False,ambiguous_abstentions=len(eligible),
                           selected_pixel_alternatives=0,physical_auto_joins_authorized=False)
        if objective-counterfactual>=margin-1e-10:chosen.append(eligible[i])
        else:abstained+=1
    if not _consistent(chosen):raise AssertionError('Conflicting hypotheses selected')
    return chosen,dict(solver_success=True,ambiguous_abstentions=abstained,
                       selected_pixel_alternatives=sum(x.origin=='s77_pixel_gap' for x in chosen),
                       physical_auto_joins_authorized=False)


def select_non_destructive(gray,polylines,baseline_edges,cutoff,arm):
    """Return same polylines, stable advisory joins, and full proposal ledger."""
    proposals,qa=alternatives_from_pixels(gray,polylines,baseline_edges,arm)
    if not qa['candidate_extra_alternatives']:
        chosen,old=select_global_forest(baseline_edges,threshold=cutoff,margin=MARGIN)
        selected_ids={id(e) for e in chosen}
        selected=[x for x in proposals if id(x.edge) in selected_ids]
        final=dict(qa,solver_success=old['solver_success'],
                   ambiguous_abstentions=old['global_ambiguous_abstentions'],
                   selected_pixel_alternatives=0,physics_auto_joins_authorized=False)
    else:
        selected,more=select_exclusive_alternatives(proposals,cutoff=cutoff)
        final={**qa,**more}
    return {'polylines':polylines,'chosen':[x.edge for x in selected],
            'selected_candidate_ids':[x.id for x in selected],
            'all_proposed_candidate_ids':[x.id for x in proposals],
            'candidate_sources':{x.id:x.origin for x in proposals},
            'candidate_ledger':proposals,'pair_count':len(baseline_edges),
            'solver_success':final['solver_success'],
            'physical_identity_verified':False,
            'physics_auto_joins_authorized':False,
            **final}
