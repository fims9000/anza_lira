"""S30 proposal graph: deduplicate near-identical candidate paths and abstain at ambiguous forks.

This module is NOT a particle classifier. All scores are uncalibrated photo
heuristics and all hypotheses are provisional even when one is preferred.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _path(p):
    x=np.asarray(p['path_points_original_xy'],dtype=float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('Invalid finite candidate centerline')
    return x


def mutual_coverage(a,b, radius=3.):
    """Fraction of each polyline near the other; symmetric via minimum."""
    if radius<=0:raise ValueError('radius must be positive')
    a=_path(a);b=_path(b)
    return float(min(np.mean(cKDTree(b).query(a)[0]<=radius),np.mean(cKDTree(a).query(b)[0]<=radius)))


def gate_candidates(candidates, *, duplicate_coverage=.87, ambiguity_margin=.08,
                    min_support=.15, max_paths=16):
    """Convert a candidate list into explainable review-only decisions.

    The score is only used to establish ordering *within* one photograph. A
    duplicate is never deleted: it is associated with a canonical hypothesis.
    A fork with two sufficiently distinct paths receives AMBIGUOUS_REVIEW.
    """
    if not 0<duplicate_coverage<=1 or ambiguity_margin<0 or not 0<=min_support<=1:
        raise ValueError('Invalid thresholds')
    if len(candidates)>max_paths:raise ValueError('too many candidate paths')
    rows=[]
    for i,c in enumerate(candidates):
        _path(c)
        val=float(c.get('model_selection_score_UNCALIBRATED',float('nan')))
        support=float(c.get('fraction_steps_above_local_signal_0_15',float('nan')))
        if not np.isfinite(val) or not np.isfinite(support) or not 0<=support<=1:
            raise ValueError('Uncalibrated score/support missing or invalid')
        rows.append({'index':i,'score':val,'support':support,'duplicate_of':None,
                     'automatic_physical_join_allowed':False,'human_confirmed_track':False})
    ranked=sorted(range(len(rows)),key=lambda i:(-rows[i]['score'],i))
    unique=[]
    for i in ranked:
        for j in unique:
            if mutual_coverage(candidates[i],candidates[j])>=duplicate_coverage:
                rows[i]['duplicate_of']=j
                break
        if rows[i]['duplicate_of'] is None:unique.append(i)
    if not unique:return {'decision':'NO_CANDIDATES','preferred_index':None,'rows':[],
                          'physical_join_authorized':False}
    first=unique[0]
    competitor=unique[1] if len(unique)>1 else None
    # Scores are not calibrated probabilities and are only within-example units.
    # The margin is a **relative** tie heuristic, not a certification bound.
    gap=(rows[first]['score']-rows[competitor]['score'])/max(abs(rows[first]['score']),abs(rows[competitor]['score']),1e-6) if competitor is not None else None
    if rows[first]['support']<min_support:state='LOW_EVIDENCE_REVIEW'
    elif competitor is not None and gap<ambiguity_margin:state='AMBIGUOUS_REVIEW'
    else:state='SINGLE_PROVISIONAL_REVIEW'
    return {'decision':state,'preferred_index':first,'competing_index':competitor,
            'relative_score_margin_NOT_CALIBRATED':gap,'unique_path_count':len(unique),
            'rows':rows,'physical_join_authorized':False,
            'expert_truth_added':0,'label_provenance':'UNKNOWN_UNVERIFIED_PHOTO_CANDIDATES'}
