"""S80: multiscale pixel-only *additional* ridge hypotheses; no oracle GT.

Original curves remain bitwise unchanged; scale-specific candidates are optional
extra instance proposals, not corrected physical trajectories. All parameters are
pre-registered in S80_FROZEN_MULTISCALE_PIXEL_INSTANCES_2026_10_26.md.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import cKDTree

from .s39_machine_auto import extract_ridges,ridge_evidence,sample_bilinear

ARMS=('baseline','s80_medium_add4','s80_consensus_add6')
CAP=32
RAD=3.0
DUP_OVERLAP=.75
CONS_MEDIUM=.60
CONS_COARSE=.40


def _coverage(a:np.ndarray,b:np.ndarray)->float:
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=2 or b.shape[1]!=2 or not len(a) or not len(b):
        raise ValueError('Invalid pixel-only 2D line geometry')
    return float((cKDTree(b).query(a)[0]<=RAD).mean())


def _duplicate(a:np.ndarray,b:np.ndarray)->bool:
    return _coverage(a,b)>=DUP_OVERLAP and _coverage(b,a)>=DUP_OVERLAP


def augment_pixel_only(gray:np.ndarray,base_lines:list[np.ndarray],arm:str)->dict:
    """Pixel-only source. Never accept synthetic track IDs, GT, or graph labels."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S80 requires 192x192 uint8 photograph-backed pixels')
    if arm not in ARMS or len(base_lines)>CAP:
        raise ValueError('Invalid arm or S39 candidate budget')
    source=list(base_lines)
    if any(not isinstance(p,np.ndarray) or p.ndim!=2 or p.shape[1]!=2 or
           len(p)<8 or not np.isfinite(p).all() for p in source):
        raise ValueError('Malformed pixel-only curve')
    result={'polylines':source.copy(),'original_count':len(source),'extra_count':0,
            'duplicate_suppressed':0,'scale_consensus_rejected':0,
            'physical_auto_joins_authorized':False,'native_GT_count':0,
            'method':arm}
    if arm=='baseline' or len(source)>=CAP:
        return result
    maps=ridge_evidence(gray)
    medium=extract_ridges({'score':maps['scales'][1], 'darkness':maps['darkness']},max_candidates=32)
    coarse=extract_ridges({'score':maps['scales'][2], 'darkness':maps['darkness']},max_candidates=32) if arm=='s80_consensus_add6' else []
    limit=4 if arm=='s80_medium_add4' else 6
    # Ranking uses photometric evidence only; never synthetic truth or proximity to injected nodes.
    medium.sort(key=lambda p:-float(np.median(sample_bilinear(maps['scales'][1],p))))
    extra=[]
    for p in medium:
        if len(extra)>=min(limit,CAP-len(source)):break
        if arm=='s80_consensus_add6' and not any(
           _coverage(p,q)>=CONS_MEDIUM and _coverage(q,p)>=CONS_COARSE for q in coarse):
            result['scale_consensus_rejected']+=1
            continue
        if any(_duplicate(p,q) for q in source+extra):
            result['duplicate_suppressed']+=1
            continue
        extra.append(p)
    result['polylines']=source+extra
    result['extra_count']=len(extra)
    assert all(a is b for a,b in zip(result['polylines'],base_lines))
    return result


def choose_cal_only(stats:dict)->dict:
    """Labels permitted only for aggregated source-CAL decision, NOT inference."""
    conditions=('clean','blur_sigma_11')
    fields=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
            'complete_three_piece_paths_synthetic','shape_true_fragments_matched','solver_failure','added_pixel_instances')
    if set(stats)!=set(ARMS) or any(set(stats[arm])!=set(conditions) for arm in ARMS):
        raise ValueError('All frozen S80 CAL arms and conditions required')
    for arm in ARMS:
        for cond in conditions:
            x=stats[arm][cond]
            if any(k not in x or type(x[k]) is not int or x[k]<0 for k in fields):
                raise ValueError('Invalid synthetic CAL aggregate')
    base=stats['baseline'];eligible=[]
    for arm in ARMS[1:]:
        clean=stats[arm]['clean'];blur=stats[arm]['blur_sigma_11']
        bc=base['clean'];bb=base['blur_sigma_11']
        if (blur['selected_synthetic_correct_links']>=bb['selected_synthetic_correct_links']+2 and
            blur['synthetic_noncorrect_selected']<=bb['synthetic_noncorrect_selected'] and
            blur['complete_three_piece_paths_synthetic']>=bb['complete_three_piece_paths_synthetic'] and
            blur['shape_true_fragments_matched']>=bb['shape_true_fragments_matched']+2 and
            clean['selected_synthetic_correct_links']>=bc['selected_synthetic_correct_links']-1 and
            clean['synthetic_noncorrect_selected']<=bc['synthetic_noncorrect_selected'] and
            clean['complete_three_piece_paths_synthetic']>=bc['complete_three_piece_paths_synthetic'] and
            clean['solver_failure']==0 and blur['solver_failure']==0):
            eligible.append(arm)
    chosen=min(eligible,key=lambda arm:(
        -stats[arm]['blur_sigma_11']['selected_synthetic_correct_links'],
        stats[arm]['blur_sigma_11']['synthetic_noncorrect_selected'],
        stats[arm]['blur_sigma_11']['added_pixel_instances'],arm)) if eligible else 'baseline'
    return {'CAL_selected':chosen,'eligible':eligible,'default_unchanged':chosen=='baseline',
            'native_physical_auto_joins_authorized':False}
