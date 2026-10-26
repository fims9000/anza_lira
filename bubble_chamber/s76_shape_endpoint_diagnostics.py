"""S76: evaluator-only endpoint-blind synthetic shape association and endpoint audit.

Absolutely NOT an inference module. It consumes synthetic generator coordinates
only *after* the image-only detector and graph solver have made predictions.
It cannot identify any genuine H50Z particle in the absence of physics GT.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .s64_pixel_to_path import INJECTED_TRUE_PAIRS
from .s69_orientation_invariant_audit import reference_facing_ports

PRED_RADIUS_PX=3.5
SOURCE_RADIUS_PX=4.5
MIN_PRED_COVERAGE=.65
MIN_SOURCE_COVERAGE=.50
AMBIGUITY_PRODUCT_MARGIN=.10
ENDPOINT_RADIUS_PX=8.
SCHEMA='S76_ENDPOINT_BLIND_SHAPE_POSTHOC_SYNTHETIC_ONLY_V1'


def _curve(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
        raise ValueError('Malformed 2D polyline in synthetic-only evaluator')
    return a


def endpoint_pair_distance(predicted, source):
    """No assumed source or predicted point-array order; does NOT assign identity."""
    a,b=_curve(predicted),_curve(source)
    d=np.linalg.norm(a[[0,-1]][:,None,:]-b[[0,-1]][None,:,:],axis=2)
    direct=(float(d[0,0]),float(d[1,1]))
    reversed_=(float(d[0,1]),float(d[1,0]))
    return min((direct,reversed_),key=lambda x:(max(x),sum(x)))


def shape_only_match(pieces, reference):
    """One-to-one synthetic-ID matching from curve support; NO endpoint features.

    The output is private evaluation state; neither used for graph predictions
    nor saved with individual native photo/crop coordinates to public GitHub.
    """
    if len(reference)!=7:raise ValueError('S56 scene must have seven artificial fragments')
    gt=[_curve(q) for q in reference]
    pred=[_curve(p) for p in pieces]
    n=len(pred)
    if n==0:return {},{'unmatched':0,'ambiguous':0,'matched':0,'matched_artificial_decoy':0},{}
    score=np.full((n,7),-1_000_000.,float)
    valid=np.zeros((n,7),bool)
    details={}
    trees=[cKDTree(b) for b in gt]
    for i,a in enumerate(pred):
        ta=cKDTree(a)
        for j,b in enumerate(gt):
            prec=float(np.mean(trees[j].query(a)[0]<=PRED_RADIUS_PX))
            recall=float(np.mean(ta.query(b)[0]<=SOURCE_RADIUS_PX))
            good=prec>=MIN_PRED_COVERAGE and recall>=MIN_SOURCE_COVERAGE
            if good:
                joint=prec*recall
                score[i,j]=1.+joint  # only geometry support, NOT endpoints or ports
                valid[i,j]=True
                details[(i,j)]={'predicted_support':prec,'source_support':recall,'joint':joint}
    ambiguous=set()
    for i in range(n):
        options=np.where(valid[i])[0]
        if len(options)>1:
            strongest=sorted((details[(i,int(j))]['joint'] for j in options),reverse=True)
            if strongest[0]-strongest[1]<=AMBIGUITY_PRODUCT_MARGIN:
                score[i,:]=-1_000_000.;valid[i,:]=False;ambiguous.add(i)
    # A no-match column per predicted polyline prevents forced identities.
    scores=np.concatenate([score,np.zeros((n,n),float)],axis=1)
    rr,cc=linear_sum_assignment(-scores)
    matched={int(i):int(j) for i,j in zip(rr,cc) if j<7 and valid[i,j]}
    audit={'matched':len(matched),'unmatched':n-len(matched),
           'ambiguous':len(ambiguous),
           'matched_artificial_decoy':int(6 in matched.values())}
    return matched,audit,{i:details[(i,j)] for i,j in matched.items()}


def _port_class(p,selected_port,source,facing_port):
    if selected_port not in (-1,0):raise ValueError('Only 2 polyline endpoint ports')
    target=_curve(source)[facing_port]
    a=_curve(p)
    current=float(np.linalg.norm(a[selected_port]-target))
    opposite=float(np.linalg.norm(a[0 if selected_port==-1 else -1]-target))
    if current<=ENDPOINT_RADIUS_PX:return 'selected_endpoint_near',current
    if opposite<=ENDPOINT_RADIUS_PX:return 'wrong_port_choice',current
    return 'endpoint_unlocalized',current


def audit_shape_vs_endpoint(inference:dict, injected:list[np.ndarray]):
    """Post-inference orthogonal diagnostics: shape identity, then endpoint ports.

    The S69 standard official evaluator is separate and is NOT monkey-patched,
    modified, or substituted with this experiment's diagnostic categories.
    """
    if len(injected)!=7:raise ValueError('Expected 7 synthetic proxy fragments')
    pieces=inference['polylines'];edges=inference['chosen']
    matched,qa,_=shape_only_match(pieces,injected)
    endpoint_good=0;endpoint_bad=0
    true_endpoint_good=true_endpoint_bad=decoy_endpoint_bad=0
    for i,j in matched.items():
        pair=endpoint_pair_distance(pieces[i],injected[j])
        if max(pair)<=ENDPOINT_RADIUS_PX:
            endpoint_good+=1
            if j<6:true_endpoint_good+=1
        else:
            endpoint_bad+=1
            if j<6:true_endpoint_bad+=1
            else:decoy_endpoint_bad+=1
    counts={k:0 for k in ('matched_neighbor_correct_port','matched_neighbor_wrong_port',
                          'matched_neighbor_endpoint_unlocalized','matched_wrong_identity_pair',
                          'unmatched_or_ambiguous_endpoint_blind')}
    correct_pairs=set();correct_edges=set()
    for e in edges:
        if not (0<=e.i<len(pieces) and 0<=e.j<len(pieces)):
            raise ValueError('Invalid machine graph endpoint index')
        a,b=matched.get(e.i),matched.get(e.j)
        if a is None or b is None:
            counts['unmatched_or_ambiguous_endpoint_blind']+=1;continue
        ids=tuple(sorted((a,b)))
        if ids not in INJECTED_TRUE_PAIRS:
            counts['matched_wrong_identity_pair']+=1;continue
        fa,fb=reference_facing_ports(injected[a],injected[b])
        cls_a,_=_port_class(pieces[e.i],e.end_i,injected[a],fa)
        cls_b,_=_port_class(pieces[e.j],e.end_j,injected[b],fb)
        if 'endpoint_unlocalized' in (cls_a,cls_b):
            counts['matched_neighbor_endpoint_unlocalized']+=1
        elif 'wrong_port_choice' in (cls_a,cls_b):
            counts['matched_neighbor_wrong_port']+=1
        else:
            counts['matched_neighbor_correct_port']+=1
            correct_pairs.add(ids)
            correct_edges.add((e.i,e.j))
    if sum(counts.values())!=len(edges):raise AssertionError('Incomplete link accounting')
    # A full synthetic three-piece chain needs BOTH correct bonds without
    # conflicting selected edges using the same inferred source components.
    full=0
    for src_ids,bonds in (((0,1,2),((0,1),(1,2))),((3,4,5),((3,4),(4,5)))):
        if not all(x in correct_pairs for x in bonds):continue
        nodes={i for i,j in matched.items() if j in src_ids}
        if len(nodes)!=3:continue
        if any((e.i in nodes or e.j in nodes) and (e.i,e.j) not in correct_edges for e in edges):continue
        full+=1
    refs=set(matched.values())
    available=sum(int(a in refs and b in refs) for a,b in INJECTED_TRUE_PAIRS)
    complete_available=int({0,1,2}<=refs)+int({3,4,5}<=refs)
    result={**counts,'predicted_curve_count':len(pieces),
         'shape_associations':qa['matched'],'shape_true_fragments_matched':sum(j<6 for j in matched.values()),
         'shape_decoy_matched':qa['matched_artificial_decoy'],
         'shape_unmatched_polylines':qa['unmatched'],
         'shape_ambiguous_polylines':qa['ambiguous'],
         'shape_matched_both_endpoints_close':endpoint_good,
         'shape_matched_one_or_both_endpoints_far':endpoint_bad,
         'shape_true_both_endpoints_close':true_endpoint_good,
         'shape_true_one_or_both_endpoints_far':true_endpoint_bad,
         'shape_decoy_one_or_both_endpoints_far':decoy_endpoint_bad,
         'shape_only_true_pair_availability':available,
         'shape_only_complete_chain_availability':complete_available,
         'shape_endpoint_separated_complete_chains':full,
         'shape_endpoint_separated_correct_links':counts['matched_neighbor_correct_port'],
         'synthetic_physics_GT_count':0,'physical_track_connections_authorized':False}
    if true_endpoint_good+true_endpoint_bad!=result['shape_true_fragments_matched']:
        raise AssertionError('True fragment endpoint accounting incomplete')
    if true_endpoint_bad+decoy_endpoint_bad!=endpoint_bad:
        raise AssertionError('True/decoy endpoint accounting incomplete')
    # No per-candidate coordinates or source identities in this report object.
    return result
