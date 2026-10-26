"""S64: blind pixel -> curve -> conservative global-path proposals.

The inference API NEVER accepts a synthetic source curve, GT label or native
particle identity. All matching to injected generator geometry happens only in
``audit_synthetic_scene`` after independent pixel-only prediction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .s35_fragment_link import GEOMETRY_FIELDS
from .s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from .s59_candidate_retrieval import filter_rows
from .s63_image_curve_variants import predict_variants

ARMS=('s39_original','junction_less')
ASSOCIATION={'pred_to_injected_radius_px':3.5,'pred_fraction':.65,
             'injected_to_pred_radius_px':4.5,'injected_fraction':.50,
             'max_each_endpoint_px':8.,'ambiguity_joint_coverage':.10}
INJECTED_TRUE_PAIRS=frozenset(((0,1),(1,2),(3,4),(4,5)))
GAP_PORTS={0:-1,1:None,2:0,3:-1,4:None,5:0}


def infer_from_pixels(gray:np.ndarray, scorer:Callable, cutoff:float, *, arm:str,
                      margin:float=.04)->dict:
    """Inference accepts only image pixels and fitted geometry scorer/policy.

    Intentionally has NO scene/proxy/GT parameter. Result is always advisory.
    """
    if arm not in ARMS:raise ValueError('Only preregistered S64 pixel detectors')
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Only uint8 192x192 synthetic-on-photo pixels')
    if not np.isfinite(cutoff) or not 0<=cutoff<=1 or not np.isfinite(margin) or margin<0:
        raise ValueError('Invalid CAL-only scoring policy')
    pieces=predict_variants(gray)[arm]
    if len(pieces)>32:raise ValueError('Candidate budget exceeded')
    if len(pieces)<3:
        return {'polylines':pieces,'pair_count':0,'chosen':[],
                'solver_success':True,'abstentions':0,'physics_auto_joins_authorized':False}
    rows=filter_rows(pair_rows(pieces),'strict_s39')
    if not rows:
        return {'polylines':pieces,'pair_count':0,'chosen':[],
                'solver_success':True,'abstentions':0,'physics_auto_joins_authorized':False}
    scores=np.asarray(scorer(rows),dtype=float)
    edges=scored_edges(rows,scores)
    chosen,qa=select_global_forest(edges,threshold=float(cutoff),margin=margin)
    if not qa['solver_success']:
        chosen=[]  # fail closed, record solver failure without GT-based fallback
    return {'polylines':pieces,'pair_count':len(rows),'chosen':chosen,
            'solver_success':bool(qa['solver_success']),
            'abstentions':int(qa['global_ambiguous_abstentions']),
            'physics_auto_joins_authorized':False}


def _candidate_associations(pieces:list[np.ndarray], reference:list[np.ndarray]):
    """Evaluation-only mapping: require WHOLE predicted piece to represent one GT.

    Do not substitute a union of unrelated image fragments for a source track.
    """
    if len(reference)!=7:raise ValueError('S64 evaluation requires seven artificial fragments')
    n=len(pieces)
    score=np.full((n,7),-1e6,dtype=float)
    valid=np.zeros((n,7),bool)
    details={}
    for i,p in enumerate(pieces):
        a=np.asarray(p,float)
        if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
            raise ValueError('Malformed image-derived polyline')
        for j,q in enumerate(reference):
            b=np.asarray(q,float)
            if b.ndim!=2 or b.shape[1]!=2 or len(b)<8 or not np.isfinite(b).all():
                raise ValueError('Malformed injected evaluation geometry')
            pred_cov=float((cKDTree(b).query(a)[0]<=ASSOCIATION['pred_to_injected_radius_px']).mean())
            src_cov=float((cKDTree(a).query(b)[0]<=ASSOCIATION['injected_to_pred_radius_px']).mean())
            direct=(float(np.linalg.norm(a[0]-b[0])),float(np.linalg.norm(a[-1]-b[-1])))
            reverse=(float(np.linalg.norm(a[0]-b[-1])),float(np.linalg.norm(a[-1]-b[0])))
            orientation=min((direct,reverse),key=lambda x:max(x))
            good=(pred_cov>=ASSOCIATION['pred_fraction'] and
                  src_cov>=ASSOCIATION['injected_fraction'] and
                  max(orientation)<=ASSOCIATION['max_each_endpoint_px'])
            if good:
                score[i,j]=1.+.5*(pred_cov+src_cov)-.005*sum(orientation)
                valid[i,j]=True
                details[(i,j)]={'pred_coverage':pred_cov,'source_coverage':src_cov,
                                'endpoints':orientation}
    ambiguous=set()
    # Equal support for two tracks at a crossing is NOT a synthetic identity.
    for i in range(n):
        candidates=np.where(valid[i])[0]
        if len(candidates)>1:
            strengths=sorted((details[(i,j)]['pred_coverage']*details[(i,j)]['source_coverage']
                              for j in candidates),reverse=True)
            if strengths[0]-strengths[1]<=ASSOCIATION['ambiguity_joint_coverage']:
                score[i,:]=-1e6;valid[i,:]=False;ambiguous.add(i)
    if not n:return {},{'ambiguous':0,'unmatched':0,'matched':0,'decoy_matched':0}
    # Each inferred curve may be unmatched; 7+N columns avoid forcing GT ties.
    weighted=np.concatenate((score,np.zeros((n,n),dtype=float)),axis=1)
    row_idx,col_idx=linear_sum_assignment(-weighted)
    matched={int(i):int(j) for i,j in zip(row_idx,col_idx) if j<7 and valid[i,j]}
    return matched,{'ambiguous':len(ambiguous),'unmatched':n-len(matched),
                    'matched':len(matched),'decoy_matched':int(6 in matched.values())}


def _synthetic_facing_port(node_idx:int, neighbor_idx:int):
    if (node_idx,neighbor_idx) in ((0,1),(1,2),(3,4),(4,5)):
        return -1
    if (node_idx,neighbor_idx) in ((1,0),(2,1),(4,3),(5,4)):
        return 0
    raise ValueError('Nonadjacent ground truth indices')


def _predicted_port_reaches_injected(p:np.ndarray, port:int, q:np.ndarray, gap_port:int)->bool:
    source_end=q[gap_port]
    return bool(np.linalg.norm(np.asarray(p[port],float)-np.asarray(source_end,float))<=
                ASSOCIATION['max_each_endpoint_px'])


def audit_synthetic_scene(inference:dict, injected:list[np.ndarray])->dict:
    """Post-hoc synthetic-only GT! NEVER call within infer_from_pixels()."""
    pieces=inference['polylines'];chosen=inference['chosen']
    association,assoc_qa=_candidate_associations(pieces,injected)
    true_edges=set();correct_pairs=set();wrong=0;unmatched=0;port_error=0
    for edge in chosen:
        a=association.get(edge.i);b=association.get(edge.j)
        if a is None or b is None:
            unmatched+=1;continue
        ids=tuple(sorted((a,b)))
        if ids not in INJECTED_TRUE_PAIRS:
            wrong+=1;continue
        pairs=((edge.i,edge.end_i,a,b),(edge.j,edge.end_j,b,a))
        if not all(_predicted_port_reaches_injected(pieces[pred],port,injected[src],
                        _synthetic_facing_port(src,other)) for pred,port,src,other in pairs):
            port_error+=1;continue
        correct_pairs.add(ids);true_edges.add((edge.i,edge.j))
    # Full path must have two correct bonds and no conflicting attached edge.
    full=0
    for src_ids,required in (((0,1,2),((0,1),(1,2))),((3,4,5),((3,4),(4,5)))):
        if not all(tuple(x) in correct_pairs for x in required):continue
        mapped_nodes={i for i,gt in association.items() if gt in src_ids}
        if len(mapped_nodes)!=3:continue
        if any((e.i in mapped_nodes or e.j in mapped_nodes) and (e.i,e.j) not in true_edges
               for e in chosen):continue
        full+=1
    correct=len(true_edges)
    assert correct+wrong+unmatched+port_error==len(chosen)
    return {'source_node_matches':len([i for i in association.values() if i<6]),
            'synthetic_decoy_matched':assoc_qa['decoy_matched'],
            'unmatched_polylines':assoc_qa['unmatched'],
            'ambiguous_polylines':assoc_qa['ambiguous'],
            'candidate_polylines':len(pieces),
            'candidate_pairs_after_strict_gate':inference['pair_count'],
            'selected_provisional_links':len(chosen),
            'selected_synthetic_correct_links':correct,
            'selected_synthetic_wrong_identified':wrong,
            'selected_synthetic_unmatched_or_ambiguous':unmatched,
            'selected_synthetic_wrong_endpoint':port_error,
            'synthetic_noncorrect_selected':len(chosen)-correct,
            'scene_any_noncorrect':int(len(chosen)>correct),
            'complete_three_piece_paths_synthetic':full,
            'solver_failure':int(not inference['solver_success']),
            'physical_verified_links':0}
