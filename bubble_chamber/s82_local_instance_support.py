"""S82 image-only local quality gate for S80 multiscale *extra* lines.

No synthetic GT, source label, graph identity or physics signal is read here.
All S72 original instances are preserved as the SAME Python objects.
"""
from __future__ import annotations

import cv2
import numpy as np

ARMS=('baseline','s82_local_q25','s82_local_q50')
QUANTILES={'s82_local_q25':.25,'s82_local_q50':.50}
SAMPLE_COUNT=31
NORMAL_OFFSETS=(3.,5.)
DARK_DELTA=.008
DARK_FULL_SCALE=.045


def _sample_curve(curve:np.ndarray):
    p=np.asarray(curve,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<8 or not np.isfinite(p).all():
        raise ValueError('Expected a finite 2D curve with >=8 samples')
    ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
    arclen=np.r_[0.,np.cumsum(ds)]
    kept=np.r_[True,ds>1.e-7]
    at=arclen[kept];pts=p[kept]
    if len(at)<2 or at[-1]<6:
        return None
    # Uniform along arclength, independent of the polygon's native vertex density.
    t=np.linspace(.04,.96,SAMPLE_COUNT)
    def at_frac(v):
        u=np.clip(v,0.,1.)*at[-1]
        return np.column_stack((np.interp(u,at,pts[:,0]),np.interp(u,at,pts[:,1])))
    xy=at_frac(t)
    direction=at_frac(t+.018)-at_frac(t-.018)
    norm=np.linalg.norm(direction,axis=1)
    if np.any(norm<1.e-6):
        return None
    # Tangent reversal negates normals but sampling uses both sides.
    perp=np.column_stack((-direction[:,1]/norm,direction[:,0]/norm))
    return xy,perp


def local_dark_stroke_support(gray:np.ndarray,curve:np.ndarray)->float:
    """Score in [0,1]; orientation independent, no full-image blur gate.

    For each of 31 centerline positions, compare photometry to symmetrical
    ±3 and ±5 pixel *normal* offsets. Out-of-image sidebands fail closed.
    """
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S82 accepts only image pixels: uint8 192x192')
    coords=_sample_curve(curve)
    if coords is None:
        return 0.
    xy,perp=coords
    all_points=[xy]
    for width in NORMAL_OFFSETS:
        all_points.extend((xy+width*perp,xy-width*perp))
    points=np.vstack(all_points)
    if np.any(points<0.) or np.any(points[:,0]>191.) or np.any(points[:,1]>191.):
        return 0.
    img=gray.astype(np.float32)/255.
    values=cv2.remap(img,points[:,0].astype(np.float32)[None,:],
                      points[:,1].astype(np.float32)[None,:],cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)[0].reshape(len(all_points),SAMPLE_COUNT)
    # Symmetric average means reversing vertices leaves the feature unchanged.
    local_dark=np.mean(values[1:],axis=0)-values[0]
    score=.55*float(np.mean(local_dark>DARK_DELTA)) + .45*float(np.mean(np.clip(local_dark/DARK_FULL_SCALE,0,1)))
    if not np.isfinite(score) or score<0. or score>1.+1e-8:
        raise AssertionError('Non-finite / unbounded local image support')
    return float(np.clip(score,0.,1.))


def thresholds_from_source_CAL(scores:list[float])->dict[str,float]:
    """Numerical percentiles only from pixel-derived SOURCE-CAL candidates.

    Lack of candidates is fail-closed (do not add lines).
    """
    a=np.asarray(scores,dtype=float)
    if a.size and (a.ndim!=1 or not np.isfinite(a).all() or np.any((a<0)|(a>1))):
        raise ValueError('Invalid CAL-only candidate scores')
    if not a.size:
        return {name:1. for name in QUANTILES}
    return {name:float(np.quantile(a,q)) for name,q in QUANTILES.items()}


def filter_extra_candidates(gray,base_lines,consensus_lines,arm,thresholds):
    """Retain source line objects; gate only S80's extra lines by pixels.

    The consensus generator has already capped and deduped lines; no new
    graph edges or track IDs are created by this module.
    """
    if arm not in ARMS or set(thresholds)!=set(QUANTILES):
        raise ValueError('Only frozen S82 arms and both numerical thresholds')
    for x in thresholds.values():
        if not np.isfinite(x) or x<0 or x>1:
            raise ValueError('Frozen CAL support threshold out of [0,1]')
    if len(base_lines)>32 or len(consensus_lines)>32 or len(consensus_lines)<len(base_lines):
        raise ValueError('S80 candidate budget/order invalid')
    if any(a is not b for a,b in zip(base_lines,consensus_lines)):
        raise ValueError('S80 did not retain original source line identity')
    additions=consensus_lines[len(base_lines):]
    if arm=='baseline':
        return {'polylines':list(base_lines),'extra_considered':0,'extra_accepted':0,
                'extra_rejected':0,'original_count':len(base_lines),'physical_auto_joins_authorized':False}
    scores=[local_dark_stroke_support(gray,p) for p in additions]
    accept=[p for p,s in zip(additions,scores) if s>=thresholds[arm]]
    out=list(base_lines)+accept
    assert len(out)<=32 and all(a is b for a,b in zip(base_lines,out))
    return {'polylines':out,'extra_considered':len(additions),'extra_accepted':len(accept),
            'extra_rejected':len(additions)-len(accept),'original_count':len(base_lines),
            'physical_auto_joins_authorized':False}


def choose_CAL_only(stats:dict)->dict:
    """Official synthetic CAL selection ONLY; never imported into inference."""
    conditions=('clean','blur_sigma_11')
    fields=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
            'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
            'solver_failure','extra_accepted')
    if set(stats)!=set(ARMS) or any(set(stats[a])!=set(conditions) for a in ARMS):
        raise ValueError('Require full frozen S82 CAL arms/conditions')
    for arm in ARMS:
        for cond in conditions:
            if any(k not in stats[arm][cond] or type(stats[arm][cond][k]) is not int or
                   stats[arm][cond][k]<0 for k in fields):
                raise ValueError('Invalid source-CAL metrics')
    base=stats['baseline'];eligible=[]
    for arm in ARMS[1:]:
        b=stats[arm]['blur_sigma_11'];bb=base['blur_sigma_11']
        c=stats[arm]['clean'];bc=base['clean']
        if (b['selected_synthetic_correct_links']>=bb['selected_synthetic_correct_links']+2 and
            b['synthetic_noncorrect_selected']<=bb['synthetic_noncorrect_selected'] and
            b['complete_three_piece_paths_synthetic']>=bb['complete_three_piece_paths_synthetic'] and
            b['shape_true_fragments_matched']>=bb['shape_true_fragments_matched']+2 and
            c['selected_synthetic_correct_links']>=bc['selected_synthetic_correct_links']-1 and
            c['synthetic_noncorrect_selected']<=bc['synthetic_noncorrect_selected'] and
            c['complete_three_piece_paths_synthetic']>=bc['complete_three_piece_paths_synthetic'] and
            b['solver_failure']==c['solver_failure']==0):
            eligible.append(arm)
    chosen=min(eligible,key=lambda x:(-stats[x]['blur_sigma_11']['selected_synthetic_correct_links'],
              stats[x]['blur_sigma_11']['synthetic_noncorrect_selected'],
              stats[x]['blur_sigma_11']['extra_accepted'],0 if x=='s82_local_q50' else 1)) if eligible else 'baseline'
    return {'CAL_selected':chosen,'eligible':eligible,'main_S72_unchanged':True,
            'no_native_H50Z_physical_joins_authorized':True}
