"""S83 pixel-only unsaturated sideband contrast + spatial consistency.

Only selectively add S80 *extra* polylines, never modify S72 source lines or
infer real particle identities. No synthetic simulator labels enter inference.
"""
from __future__ import annotations
import cv2
import numpy as np
from .s82_local_instance_support import _sample_curve,SAMPLE_COUNT,NORMAL_OFFSETS

ARMS=('baseline','s83_local_q40','s83_local_q70')
QUANTILES={'s83_local_q40':.40,'s83_local_q70':.70}
CONSISTENCY_DENOM_FLOOR=.005


def local_dark_stroke_support(gray:np.ndarray,curve:np.ndarray)->float:
    """Continuous [0,1] dark-stroke support; invariant to vertex reversal."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S83 sees only 192x192 uint8 pixels')
    sample=_sample_curve(curve)
    if sample is None:return 0.
    xy,normal=sample
    coords=[xy]
    for d in NORMAL_OFFSETS:
        coords.extend((xy+d*normal,xy-d*normal))
    all_points=np.vstack(coords)
    if (np.any(all_points<0) or np.any(all_points[:,0]>191) or np.any(all_points[:,1]>191)):
        return 0.
    img=gray.astype(np.float32)/255.
    vals=cv2.remap(img,all_points[:,0].astype('float32')[None,:],
                   all_points[:,1].astype('float32')[None,:],cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)[0].reshape(len(coords),SAMPLE_COUNT)
    signal=np.maximum(np.mean(vals[1:],axis=0)-vals[0],0.)
    consistency=float(np.quantile(signal,.25)/(np.quantile(signal,.75)+CONSISTENCY_DENOM_FLOOR))
    score=float(np.mean(signal))*(.35+.65*consistency)
    if not np.isfinite(score) or score<0 or score>1+1e-7:
        raise AssertionError('Unbounded S83 photometric feature')
    return float(np.clip(score,0,1))


def thresholds_from_source_CAL(scores:list[float])->dict[str,float]:
    a=np.asarray(scores,dtype=float)
    if a.size and (a.ndim!=1 or not np.isfinite(a).all() or np.any((a<0)|(a>1))):
        raise ValueError('Invalid source CAL feature')
    return {name:float(np.quantile(a,q)) if a.size else 1.0 for name,q in QUANTILES.items()}


def filter_extra_candidates(gray,base_lines,consensus_lines,arm,thresholds):
    if arm not in ARMS or set(thresholds)!=set(QUANTILES):
        raise ValueError('S83 frozen arm or source-CAL thresholds missing')
    if any(not np.isfinite(v) or v<0 or v>1 for v in thresholds.values()):
        raise ValueError('CAL pixel-only threshold not in [0,1]')
    if len(base_lines)>32 or len(consensus_lines)>32 or len(consensus_lines)<len(base_lines):
        raise ValueError('S80 original line budget/order violation')
    if any(a is not b for a,b in zip(base_lines,consensus_lines)):
        raise ValueError('Original source line objects are not preserved')
    extras=consensus_lines[len(base_lines):]
    if arm=='baseline':
        return {'polylines':list(base_lines),'extra_considered':0,'extra_accepted':0,
                'extra_rejected':0,'original_count':len(base_lines),'physical_auto_joins_authorized':False}
    scores=[local_dark_stroke_support(gray,p) for p in extras]
    accepted=[p for p,s in zip(extras,scores) if s>=thresholds[arm]]
    lines=list(base_lines)+accepted
    assert len(lines)<=32 and all(a is b for a,b in zip(base_lines,lines))
    return {'polylines':lines,'extra_considered':len(extras),'extra_accepted':len(accepted),
            'extra_rejected':len(extras)-len(accepted),'original_count':len(base_lines),
            'physical_auto_joins_authorized':False}


def choose_CAL_only(stats):
    conditions=('clean','blur_sigma_11')
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
          'solver_failure','extra_accepted')
    if set(stats)!=set(ARMS) or any(set(stats[a])!=set(conditions) for a in ARMS):
        raise ValueError('S83 requires full CAL arm/condition grid')
    for a in ARMS:
        for cond in conditions:
            if any(k not in stats[a][cond] or type(stats[a][cond][k]) is not int or
                   stats[a][cond][k]<0 for k in keys):
                raise ValueError('Malformed source CAL aggregated metrics')
    base=stats['baseline'];eligible=[]
    for a in ARMS[1:]:
        b=stats[a]['blur_sigma_11'];bb=base['blur_sigma_11']
        c=stats[a]['clean'];bc=base['clean']
        if (b['selected_synthetic_correct_links']>=bb['selected_synthetic_correct_links']+2 and
            b['synthetic_noncorrect_selected']<=bb['synthetic_noncorrect_selected'] and
            b['complete_three_piece_paths_synthetic']>=bb['complete_three_piece_paths_synthetic'] and
            b['shape_true_fragments_matched']>=bb['shape_true_fragments_matched']+2 and
            c['selected_synthetic_correct_links']>=bc['selected_synthetic_correct_links']-1 and
            c['synthetic_noncorrect_selected']<=bc['synthetic_noncorrect_selected'] and
            c['complete_three_piece_paths_synthetic']>=bc['complete_three_piece_paths_synthetic'] and
            b['solver_failure']==c['solver_failure']==0):
            eligible.append(a)
    winner=min(eligible,key=lambda name:(-stats[name]['blur_sigma_11']['selected_synthetic_correct_links'],
            stats[name]['blur_sigma_11']['synthetic_noncorrect_selected'],
            stats[name]['blur_sigma_11']['extra_accepted'],0 if name=='s83_local_q70' else 1)) if eligible else 'baseline'
    return {'CAL_selected':winner,'eligible':eligible,'main_S72_unchanged':True,
            'physical_auto_joins_authorized':False}
