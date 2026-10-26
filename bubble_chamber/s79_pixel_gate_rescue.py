"""S79 pixel-only rescue of S59-rejected *pair candidates*, not native tracks.

Every pre-existing line is preserved verbatim. Only S77-supported short-gap
pairs may bypass a strict geometry gate, with all original candidates retained.
No artificial source identities/coordinates are accepted by inference.
"""
from __future__ import annotations
import numpy as np
from .s56_global_path_assembly import Edge
from .s77_short_gap_instances import propose_pixel_gap_pairs
from .s78_exclusive_alternative_graph import (Candidate,select_exclusive_alternatives,
    MARGIN,MAX_EXTRA)

ARMS=('baseline','rescue_strict','rescue_relaxed')
PROPOSAL_SOURCE={'rescue_strict':'short_gap_strict','rescue_relaxed':'short_gap_relaxed'}


def additional_candidates(gray,polylines,strict_edges,complete_rows,complete_scores,arm):
    if arm not in ARMS or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Invalid fixed S79 pixel arm')
    if len(polylines)>32 or len(complete_rows)!=len(complete_scores):
        raise ValueError('Bad input candidate dimensions')
    originals={(e.i,e.j):e for e in strict_edges}
    complete={(int(row['i']),int(row['j'])):float(score)
              for row,score in zip(complete_rows,complete_scores)}
    proposals=[Candidate(e,'s56_geometry',f'S79-G-{i}-{e.i}-{e.j}')
               for i,e in enumerate(strict_edges)]
    qa={'original_polylines_retained':len(polylines),'original_pair_candidates_retained':len(strict_edges),
        'raw_S77_gap_hypotheses':0,'duplicates_same_ports_suppressed':0,
        'rescued_from_strict_pair_gate':0,'different_port_competitors':0,
        'pixel_alternatives_added':0}
    if arm=='baseline' or len(polylines)<2:return proposals,qa
    candidates=propose_pixel_gap_pairs(gray,polylines,PROPOSAL_SOURCE[arm])
    qa['raw_S77_gap_hypotheses']=len(candidates)
    for local_score,a,b in candidates:
        (i,pi),(j,pj)=a,b
        if i==j:continue
        if i>j:i,j,pi,pj=j,i,pj,pi
        key=(i,j)
        if key not in complete:raise AssertionError('S56 geometry features absent for S77 pair')
        base=originals.get(key)
        if base is not None and (base.end_i,base.end_j)==(pi,pj):
            qa['duplicates_same_ports_suppressed']+=1;continue
        if not np.isfinite(local_score) or not 0<=local_score<=1:
            raise ValueError('Invalid local pixel gap support')
        scoring=float(complete[key])
        if not np.isfinite(scoring):raise ValueError('Nonfinite original geometry score')
        rank=min(.999,max(scoring+.06,.68+.24*float(local_score)))
        edge=Edge(i,j,int(pi),int(pj),rank)
        proposals.append(Candidate(edge,'s77_pixel_gap',f'S79-P-{arm}-{i}-{j}-{pi}-{pj}'))
        qa['pixel_alternatives_added']+=1
        if base is None:qa['rescued_from_strict_pair_gate']+=1
        else:qa['different_port_competitors']+=1
        if qa['pixel_alternatives_added']>MAX_EXTRA:raise ValueError('Too many local proposals')
    if len({p.id for p in proposals})!=len(proposals):raise AssertionError('Duplicate candidate IDs')
    return proposals,qa


def solve_s79_from_pixels(gray,polylines,strict_edges,complete_rows,complete_scores,
                          cutoff,arm,baseline_chosen=None,baseline_success=True):
    proposals,qa=additional_candidates(gray,polylines,strict_edges,complete_rows,complete_scores,arm)
    if qa['pixel_alternatives_added']==0:
        if baseline_chosen is None:raise ValueError('Original baseline proposal must be supplied')
        chosen_edges=list(baseline_chosen)
        chosen_ids={id(e) for e in chosen_edges}
        chosen=[p for p in proposals if id(p.edge) in chosen_ids]
        final={'solver_success':bool(baseline_success),'ambiguous_abstentions':0,
               'selected_pixel_alternatives':0,'physical_auto_joins_authorized':False}
    else:
        chosen,final=select_exclusive_alternatives(proposals,cutoff=cutoff,margin=MARGIN)
        chosen_edges=[p.edge for p in chosen]
    if len(chosen_edges)>max(0,len(polylines)-1):raise AssertionError('Cycle/forest bug')
    return {'polylines':polylines,'chosen':chosen_edges,
            'pair_count':len(strict_edges),'solver_success':final['solver_success'],
            'selected_candidate_ids':[p.id for p in chosen],
            'all_candidate_ids':[p.id for p in proposals],
            'provenance':{p.id:p.origin for p in proposals},
            'native_physical_auto_joins_authorized':False,
            **qa,**final}
