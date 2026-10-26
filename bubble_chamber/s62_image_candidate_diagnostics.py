"""S62: image-only detector output vs generator-only proxy curves.

The detector never receives source-derived source polylines; they are accepted
only by these *evaluation* functions. H50Z native physics GT remains missing.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .s39_machine_auto import ridge_evidence,contrast_baseline,canny_score,fixed_budget_mask,extract_ridges

ACTIVATION_BUDGET=.035
MASK_RADIUS=2
MIN_FRAGMENT_SUPPORT=.40
IMAGE_PIECE_DISTANCE_PX=4.


def image_only_predictions(gray:np.ndarray)->dict:
    """Single-image input; no labels, source nodes, or physical priors passed."""
    if gray.ndim!=2 or gray.dtype!=np.uint8 or gray.shape!=(192,192):
        raise ValueError('Need a 192x192 synthetic-on-H50Z uint8 single image')
    ridge=ridge_evidence(gray)
    raw={'multiscale_ridge':ridge['score'],'local_darkness':contrast_baseline(gray),
         'sobel_edge':canny_score(gray)}
    masks={name:fixed_budget_mask(arr,fraction=ACTIVATION_BUDGET) for name,arr in raw.items()}
    lines=extract_ridges(ridge,max_candidates=32)
    return {'masks':masks,'image_only_ridge_polylines':lines}


def proxy_mask_fragment_support(mask:np.ndarray,polyline:np.ndarray)->dict:
    """Generator polyline appears ONLY here, after image-only prediction."""
    if mask.dtype!=bool or mask.ndim!=2:raise ValueError('Bad image-only mask')
    a=np.asarray(polyline,float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
        raise ValueError('Invalid evaluation proxy curve')
    if np.any(a<0) or np.any(a[:,0]>=mask.shape[1]) or np.any(a[:,1]>=mask.shape[0]):
        raise ValueError('Evaluation proxy outside image')
    r=MASK_RADIUS
    evidence=cv2.dilate(mask.astype('uint8'),np.ones((2*r+1,2*r+1),'uint8'))>0
    x=np.clip(np.rint(a[:,0]).astype(int),0,mask.shape[1]-1)
    y=np.clip(np.rint(a[:,1]).astype(int),0,mask.shape[0]-1)
    covered=evidence[y,x]
    ratio=float(np.mean(covered))
    return {'coverage_2px':ratio,'both_endpoints_supported_2px':bool(covered[0] and covered[-1]),
            'usable_proxy_fragment':bool(ratio>=MIN_FRAGMENT_SUPPORT and covered[0] and covered[-1])}


def proxy_extracted_piece_support(pieces:list[np.ndarray],polyline:np.ndarray)->dict:
    """Image detector output points are compared to generator proxies ONLY for QA."""
    a=np.asarray(polyline,float)
    if a.ndim!=2 or a.shape[1]!=2 or len(a)<8 or not np.isfinite(a).all():
        raise ValueError('Invalid proxy')
    if not pieces:
        return {'coverage_4px':0.,'both_endpoints_supported_4px':False,
                'usable_proxy_fragment':False}
    detected=np.concatenate([np.asarray(x,float) for x in pieces],axis=0)
    distances=cKDTree(detected).query(a)[0]
    covered=distances<=IMAGE_PIECE_DISTANCE_PX
    return {'coverage_4px':float(covered.mean()),
            'both_endpoints_supported_4px':bool(covered[0] and covered[-1]),
            'usable_proxy_fragment':bool(covered.mean()>=MIN_FRAGMENT_SUPPORT and covered[0] and covered[-1])}


def scene_upper_bound_from_proxy_support(evaluated_nodes:list[dict])->dict:
    """Artificial identity known ONLY to evaluator (nodes 0-5, decoy 6)."""
    if len(evaluated_nodes)!=7 or not all('usable_proxy_fragment' in x for x in evaluated_nodes):
        raise ValueError('Require exactly seven artificially generated proxy fragments')
    usable=[bool(x['usable_proxy_fragment']) for x in evaluated_nodes]
    pairs=((0,1),(1,2),(3,4),(4,5))
    available=sum(bool(usable[i] and usable[j]) for i,j in pairs)
    full=int(all(usable[:3]))+int(all(usable[3:6]))
    return {'proxy_true_fragments_usable':sum(usable[:6]),
            'synthetic_decoy_fragment_usable':int(usable[6]),
            'proxy_true_joins_upper_bound':available,
            'proxy_complete_three_piece_paths_upper_bound':full,
            'no_native_physics_GT':True}
