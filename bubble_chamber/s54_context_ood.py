"""S54: photo-context parity and calibration-only simulator OOD warnings."""
from __future__ import annotations

import numpy as np
from .s43_artifact_triage import FEATURE_NAMES as OLD_NAMES, path_features
from .s53_bead_texture import FEATURE_NAMES as NEW_NAMES, texture_features, merge_features
from .s53_native_texture_advisory import _linear_predict
from .s39_machine_auto import ridge_evidence

SCHEMA='s54_context_aligned_machine_unknown_only_v1'
CROP_SIZE=192
MARGIN=12


def source_context_crop(gray:np.ndarray, points:np.ndarray|list,
                        *,size:int=CROP_SIZE,margin:int=MARGIN):
    """Lossless source 192px window containing all points+12px margin, or None.

    Crucial: never resize the path to fit; changes to source feature scale
    must not be mistaken for independent native physical evidence.
    """
    p=np.asarray(points,dtype=float)
    if (not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.ndim!=2
        or min(gray.shape)<size or p.ndim!=2 or p.shape[1]!=2 or len(p)<3
        or not np.isfinite(p).all() or np.any(p<0)
        or np.any(p[:,0]>=gray.shape[1]) or np.any(p[:,1]>=gray.shape[0])):
        raise ValueError('Invalid authentic photograph/candidate geometry')
    offsets=[]
    for axis,dim in enumerate((gray.shape[1],gray.shape[0])):
        mini=float(np.min(p[:,axis])); maxi=float(np.max(p[:,axis]))
        lower=max(0,int(np.ceil(maxi-(size-margin))))
        upper=min(dim-size,int(np.floor(mini-margin)))
        if lower>upper:
            return None
        prefer=int(np.rint((mini+maxi)/2.-size/2.))
        offsets.append(int(np.clip(prefer,lower,upper)))
    x0,y0=offsets
    roi=gray[y0:y0+size,x0:x0+size]
    local=p-np.array([x0,y0])
    if roi.shape!=(size,size) or np.any(local<margin-1.e-6) or np.any(local>size-margin+1.e-6):
        raise AssertionError('Context crop violates margin guarantee')
    if not np.array_equal(roi,gray[y0:y0+size,x0:x0+size]):
        raise AssertionError('Source pixel data changed')
    return roi,local,(x0,y0)


def calibration_ood_threshold(x:np.ndarray, checkpoint:dict, *,quantile:float=.99)->float:
    mu=np.asarray(checkpoint['scaler_mean'],dtype=float)
    sd=np.asarray(checkpoint['scaler_scale'],dtype=float)
    x=np.asarray(x,dtype=float)
    if (x.ndim!=2 or len(x)<20 or x.shape[1]!=len(mu)
        or len(mu)!=len(sd) or not np.isfinite(x).all()
        or not np.isfinite(mu).all() or not np.isfinite(sd).all()
        or np.any(sd<=0) or not .90<=quantile<1.):
        raise ValueError('Insufficient provenance-locked synthetic calibration')
    metric=np.max(np.abs((x-mu)/sd),axis=1)
    return float(np.quantile(metric,quantile))


def score_context_aligned(gray:np.ndarray,candidates:list[dict],*,photo:str,
                          checkpoints:dict[str,dict], thresholds:dict[str,float]) -> list[dict]:
    if photo not in ('005','006') or len(candidates)!=len({c['draft_id'] for c in candidates}):
        raise ValueError('Invalid candidate/source identity')
    source='006' if photo=='005' else '005'
    results=[]
    for c in candidates:
        crop=source_context_crop(gray,c['points_xy'])
        if crop is None:
            results.append({'draft_id':c['draft_id'],
              'status':'UNKNOWN_UNSCORABLE_CONTEXT','cannot_fit_192px_without_resampling':True,
              'native_particle_identity':'UNKNOWN','candidate_preserved':True,
              'physical_auto_join_authorized':False})
            continue
        roi,pts,(x0,y0)=crop
        maps=ridge_evidence(roi)
        x43=path_features(roi,pts,maps=maps)[None,:]
        x53=merge_features(x43,texture_features(roi,pts)[None,:])
        model43=checkpoints['existing_S43'];model53=checkpoints['S43_plus_bead_texture_S53']
        a=float(_linear_predict(x43,model43,source,photo,False)[0])
        b=float(_linear_predict(x53,model53,source,photo,True)[0])
        def novelty(arr,model):
            mu=np.asarray(model['scaler_mean'],dtype=float)
            sd=np.asarray(model['scaler_scale'],dtype=float)
            zz=np.abs((arr[0]-mu)/sd)
            idx=int(np.argmax(zz))
            return float(zz[idx]),model['feature_names'][idx]
        z43,worst43=novelty(x43,model43);z53,worst53=novelty(x53,model53)
        ood43=z43>float(thresholds['existing_S43'])
        ood53=z53>float(thresholds['S43_plus_bead_texture_S53'])
        results.append({'draft_id':c['draft_id'],
          'status':'UNKNOWN_OUTSIDE_CALIBRATION_SUPPORT' if (ood43 or ood53) else 'UNKNOWN_WITHIN_SIMULATOR_SUPPORT_ONLY',
          'S43_simulated_origin_score_192px_NOT_TRACK_PROB':round(a,6),
          'S53_simulated_origin_score_192px_NOT_TRACK_PROB':round(b,6),
          'S43_calibration_max_abs_z':round(z43,5),
          'S53_calibration_max_abs_z':round(z53,5),
          'S43_dominant_outlier_feature':worst43,
          'S53_dominant_outlier_feature':worst53,
          'S43_ood_by_calibration':bool(ood43),'S53_ood_by_calibration':bool(ood53),
          'private_context_top_left_xy':[x0,y0],
          'native_particle_identity':'UNKNOWN','candidate_preserved':True,
          'physical_auto_join_authorized':False})
    return results
