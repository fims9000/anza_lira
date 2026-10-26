"""S69 EVALUATION-ONLY fix: reference endpoint order is not particle direction.

S56 synthetic scene generation may reverse stored coordinates independently for
all seven injected fragments. The historical S64 synthetic endpoint audit
assumed a fixed array orientation. It can mark an otherwise correct 2-D bond
as wrong solely because the reference array was reversed. This module must
NEVER be imported by a production/inference path: it consumes *synthetic*
source geometry and identities only after all pixel-only predictions finish.
"""
from __future__ import annotations

import numpy as np
from .s64_pixel_to_path import (
    _candidate_associations, INJECTED_TRUE_PAIRS, ASSOCIATION,
)


def reference_facing_ports(a:np.ndarray, b:np.ndarray)->tuple[int,int]:
    """Return source-array endpoints nearest the other known true fragment.

    Only for evaluation of artificial known joins. Never determines physical
    direction or particle connectivity for native photographs.
    """
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1:]!=(2,) or b.shape[1:]!=(2,) or len(a)<2 or len(b)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Malformed synthetic evaluation geometry')
    d=np.linalg.norm(a[[0,-1]][:,None,:]-b[[0,-1]][None,:,:],axis=2)
    k=int(np.argmin(d))
    return ((0,-1)[k//2],(0,-1)[k%2])


def audit_orientation_invariant_synthetic(inference:dict, injected:list[np.ndarray])->dict:
    """Post-hoc synthetic-only scoring; does not affect model predictions."""
    if len(injected)!=7:
        raise ValueError('Frozen S56 simulation requires seven artificial fragments')
    pieces=inference['polylines'];chosen=inference['chosen']
    association,qa=_candidate_associations(pieces,injected)
    true_edges=set();correct_pairs=set()
    wrong=unmatched=port_error=0
    for edge in chosen:
        a=association.get(edge.i);b=association.get(edge.j)
        if a is None or b is None:
            unmatched+=1;continue
        ids=tuple(sorted((a,b)))
        if ids not in INJECTED_TRUE_PAIRS:
            wrong+=1;continue
        facing_a,facing_b=reference_facing_ports(injected[a],injected[b])
        if (np.linalg.norm(pieces[edge.i][edge.end_i]-injected[a][facing_a]) > ASSOCIATION['max_each_endpoint_px'] or
            np.linalg.norm(pieces[edge.j][edge.end_j]-injected[b][facing_b]) > ASSOCIATION['max_each_endpoint_px']):
            port_error+=1;continue
        correct_pairs.add(ids);true_edges.add((edge.i,edge.j))
    full=0
    for src_ids, required in (((0,1,2),((0,1),(1,2))),((3,4,5),((3,4),(4,5)))):
        if not all(edge in correct_pairs for edge in required):continue
        mapped_nodes={i for i,gt in association.items() if gt in src_ids}
        if len(mapped_nodes)!=3:continue
        if any((e.i in mapped_nodes or e.j in mapped_nodes) and (e.i,e.j) not in true_edges
               for e in chosen):continue
        full+=1
    correct=len(true_edges)
    if correct+wrong+unmatched+port_error!=len(chosen):
        raise AssertionError('Evaluation accounting not exhaustive')
    return dict(
        source_node_matches=sum(j<6 for j in association.values()),
        synthetic_decoy_matched=qa['decoy_matched'],
        unmatched_polylines=qa['unmatched'],ambiguous_polylines=qa['ambiguous'],
        candidate_polylines=len(pieces),
        candidate_pairs_after_strict_gate=inference['pair_count'],
        selected_provisional_links=len(chosen),
        selected_synthetic_correct_links=correct,
        selected_synthetic_wrong_identified=wrong,
        selected_synthetic_unmatched_or_ambiguous=unmatched,
        selected_synthetic_wrong_endpoint=port_error,
        synthetic_noncorrect_selected=len(chosen)-correct,
        scene_any_noncorrect=int(len(chosen)>correct),
        complete_three_piece_paths_synthetic=full,
        solver_failure=int(not inference['solver_success']),
        physical_verified_links=0,
        orientation_invariant_synthetic_only=True,
    )
