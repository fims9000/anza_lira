"""S81 image-only quality gate for optional S80 scale-consensus instances.

Never accesses source-truth lines, physical event IDs or any physics labels.
CAL receives the image-only score distribution from source-clean scenes.
"""
from __future__ import annotations
import cv2
import numpy as np

ARMS=('baseline','s81_gate_q20','s81_gate_q40')
QUANTILES={'s81_gate_q20':.20,'s81_gate_q40':.40}
SIGMA_FINE=.85
SIGMA_COARSE=2.6
DENOM_FLOOR=.0001


def fine_coarse_ratio(gray:np.ndarray)->float:
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Quality-gate inference requires 192x192 uint8 image pixels')
    a=gray.astype(np.float32)/255.
    fine=cv2.GaussianBlur(a,(0,0),SIGMA_FINE)
    coarse=cv2.GaussianBlur(a,(0,0),SIGMA_COARSE)
    value=float(np.mean(np.abs(a-fine))/(np.mean(np.abs(fine-coarse))+DENOM_FLOOR))
    if not np.isfinite(value) or value<0:raise ValueError('Invalid machine-only texture statistic')
    return value


def thresholds_from_CAL_clean(clean_scores:list[float])->dict:
    if len(clean_scores)<16 or any(not np.isfinite(x) or x<0 for x in clean_scores):
        raise ValueError('Need the complete clean-image source CAL feature set')
    return {arm:float(np.quantile(clean_scores,q)) for arm,q in QUANTILES.items()}


def activated(score:float,threshold:float)->bool:
    if not np.isfinite(score) or score<0 or not np.isfinite(threshold) or threshold<0:
        raise ValueError('Non-finite quality score or frozen CAL threshold')
    return bool(score<=threshold)


def choose_cal_only(stats:dict)->dict:
    conditions=('clean','blur_sigma_11')
    needed=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
            'shape_true_fragments_matched','complete_three_piece_paths_synthetic',
            'solver_failure','quality_gate_triggered')
    if set(stats)!=set(ARMS) or any(set(stats[a])!=set(conditions) for a in ARMS):
        raise ValueError('All S81 frozen variants and conditions needed')
    for arm in ARMS:
        for condition in conditions:
            row=stats[arm][condition]
            if any(key not in row or type(row[key]) is not int or row[key]<0 for key in needed):
                raise ValueError('Bad CAL synthetic-only aggregate')
    b=stats['baseline'];eligible=[]
    for arm in ARMS[1:]:
        c=stats[arm]['clean'];s=stats[arm]['blur_sigma_11']
        bc=b['clean'];bs=b['blur_sigma_11']
        if (s['selected_synthetic_correct_links']>=bs['selected_synthetic_correct_links']+2 and
            s['synthetic_noncorrect_selected']<=bs['synthetic_noncorrect_selected'] and
            s['shape_true_fragments_matched']>=bs['shape_true_fragments_matched']+2 and
            s['complete_three_piece_paths_synthetic']>=bs['complete_three_piece_paths_synthetic'] and
            c['selected_synthetic_correct_links']>=bc['selected_synthetic_correct_links']-1 and
            c['synthetic_noncorrect_selected']<=bc['synthetic_noncorrect_selected'] and
            c['complete_three_piece_paths_synthetic']>=bc['complete_three_piece_paths_synthetic'] and
            c['solver_failure']==s['solver_failure']==0):
            eligible.append(arm)
    selected=min(eligible,key=lambda a:(
          -stats[a]['blur_sigma_11']['selected_synthetic_correct_links'],
          stats[a]['blur_sigma_11']['synthetic_noncorrect_selected'],
          stats[a]['clean']['quality_gate_triggered'],a)) if eligible else 'baseline'
    return {'CAL_selected':selected,'eligible':eligible,
            'original_S72_remains_default':selected=='baseline',
            'native_physical_auto_join_authorized':False}
