"""S74 fixed pixel-only sharpening controls; no simulator labels or GT."""
from __future__ import annotations
import cv2
import numpy as np
ARMS=('identity','unsharp_065','unsharp_130')
AMOUNTS={'identity':0.,'unsharp_065':.65,'unsharp_130':1.30}


def preprocess(gray:np.ndarray,arm:str)->np.ndarray:
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S74 requires only uint8 192x192 pixel input')
    if arm not in ARMS:raise ValueError('Only frozen preprocessing arms')
    if arm=='identity':return gray.copy()
    blurred=cv2.GaussianBlur(gray,(0,0),sigmaX=1.1,sigmaY=1.1)
    amount=AMOUNTS[arm]
    return np.clip(np.rint(gray.astype('float32')*(1.+amount)-blurred.astype('float32')*amount),0,255).astype('uint8')


def choose_cal_only(cal:dict[str,dict[str,dict[str,int]]])->dict:
    if set(cal)!=set(ARMS):raise ValueError('Full CAL method grid required')
    names=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
           'complete_three_piece_paths_synthetic','solver_failure')
    if any(set(v)!=set(('clean','blur_sigma_11')) for v in cal.values()):
        raise ValueError('Need CAL results on matched clean AND blurred images')
    for val in cal.values():
        for x in val.values():
            if any(k not in x or type(x[k]) is not int or x[k]<0 for k in names):
                raise ValueError('Invalid synthetic CAL metrics')
    base=cal['identity']
    b0,b1=base['clean'],base['blur_sigma_11']
    candidates=[]
    for arm in ARMS[1:]:
        c,v=cal[arm]['clean'],cal[arm]['blur_sigma_11']
        if (v['selected_synthetic_correct_links']>=b1['selected_synthetic_correct_links']+2 and
            v['synthetic_noncorrect_selected']<=b1['synthetic_noncorrect_selected'] and
            v['complete_three_piece_paths_synthetic']>=b1['complete_three_piece_paths_synthetic'] and
            c['selected_synthetic_correct_links']>=b0['selected_synthetic_correct_links']-2 and
            c['synthetic_noncorrect_selected']<=b0['synthetic_noncorrect_selected']+2 and
            c['complete_three_piece_paths_synthetic']>=b0['complete_three_piece_paths_synthetic']-1 and
            c['solver_failure']==0 and v['solver_failure']==0):
            candidates.append(arm)
    chosen=min(candidates,key=lambda a:(-cal[a]['blur_sigma_11']['selected_synthetic_correct_links'],
        cal[a]['blur_sigma_11']['synthetic_noncorrect_selected'],
        base['clean']['selected_synthetic_correct_links']-cal[a]['clean']['selected_synthetic_correct_links'],
        ARMS.index(a))) if candidates else 'identity'
    return {'chosen_CAL_only':chosen,'eligible_arms':candidates,
      'default_remains_original_S72':chosen=='identity','native_physics_join_authority':False}
