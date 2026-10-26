"""S42. Geometric-only machine pseudo teacher, no cross-photo AZ/UNet leakage.

Pseudo labels 0 and 1 are ASSUMPTIONS, not native particle truth. Use only for
machine pretraining, never for expert evaluation or physical particle claims.
"""
from __future__ import annotations

import cv2
import numpy as np

from .s39_machine_auto import ridge_evidence,extract_ridges,line_evidence
from .s41_machine_quality import uncertainty_region,refine_pseudo_mask

SCHEMA='s42_geometry_only_nonoracle_pseudo_teacher_v1'


def geometry_only_pseudo(gray:np.ndarray,zones:list[dict],origin_xy:list,photo:str)->dict:
    if gray.shape!=(512,512) or gray.dtype!=np.uint8:
        raise ValueError('S42 expects full 512x512 H50Z tile')
    evidence=ridge_evidence(gray)
    curves=extract_ridges(evidence,max_candidates=32)
    proposed=[];machine_positives=[]
    for k,path in enumerate(curves):
        cues=line_evidence(gray,path,evidence,az_map=None,prior_drafts=[])
        row={'draft_id':f's42_pure_geometry_{k:03d}',
             'source':'Hessian_only; no neural teacher',
             'points_xy':np.round(path,2).tolist()}
        proposed.append(row)
        if (cues['raw_center_minus_flank_8bit']>=8 and
                cues['ridge_scale_consensus_fraction']>=.34):
            machine_positives.append(path)
    old=np.full(gray.shape,255,dtype=np.uint8)
    pos=np.zeros(gray.shape,dtype=np.uint8)
    proposed_pixels=np.zeros_like(pos)
    for path in machine_positives:
        cv2.polylines(pos,[np.rint(path).astype(np.int32)],False,1,2)
    for path in curves:
        cv2.polylines(proposed_pixels,[np.rint(path).astype(np.int32)],False,1,5)
    old[pos!=0]=1
    if machine_positives:
        near=(cv2.dilate(pos,np.ones((31,31),np.uint8))>0)
        close=(cv2.dilate(pos,np.ones((21,21),np.uint8))>0)
        outside_all=cv2.dilate(proposed_pixels,np.ones((11,11),np.uint8))==0
        quiet=evidence['score']<np.percentile(evidence['score'],75)
        old[near & ~close & outside_all & quiet]=0
    # Only the known fiducial controls may downgrade pseudo labels to unknown.
    # The negative S41 synthetic scratch experiment vetoed photometric gating.
    dummy=[{'photometric_machine_stable_NOT_GT':True} for _ in proposed]
    protected,flags=uncertainty_region(gray.shape,proposed,dummy,zones,origin_xy,photo)
    mask,stats=refine_pseudo_mask(old,protected)
    return {'schema':SCHEMA,'mask':mask,'ridge_count':len(curves),
        'pseudo_positive_curves':len(machine_positives),
        'fiducial_crossing_curves':sum(f['fiducial_risk'] for f in flags),
        'pixel_change':stats,'AZ_model_used_in_teacher':False,
        'S31_machine_draft_used_in_teacher':False,
        'native_expert_gt_used':False}
