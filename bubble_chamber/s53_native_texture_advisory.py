"""S53 native H50Z machine advisory with strict synthetic/physics separation.

This module NEVER returns a physical track prediction, confidence, hard label,
or an authorized physical join. Results are renderer-origin similarities only.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .s39_machine_auto import ridge_evidence
from .s43_artifact_triage import FEATURE_NAMES as OLD_NAMES, path_features
from .s53_bead_texture import FEATURE_NAMES as NEW_NAMES, SCHEMA, texture_features, merge_features


def _linear_predict(x:np.ndarray, checkpoint:dict, expected_source_photo:str,
                    target_photo:str, extended:bool) -> np.ndarray:
    cols=list(OLD_NAMES)+(list(NEW_NAMES) if extended else [])
    arm='S43_plus_bead_texture_S53' if extended else 'existing_S43'
    if (checkpoint.get('schema')!=SCHEMA or checkpoint.get('synthetic_only') is not True
        or checkpoint.get('arm')!=arm
        or checkpoint.get('feature_names')!=cols
        or checkpoint.get('source_photo')!=expected_source_photo
        or checkpoint.get('test_photo')!=target_photo
        or target_photo==expected_source_photo
        or checkpoint.get('native_physics_classes_known') is not False
        or checkpoint.get('native_auto_join_allowed') is not False):
        raise ValueError('Untrusted source-provenance or unsafe synthetic origin model')
    x=np.asarray(x,dtype=float)
    mu=np.asarray(checkpoint['scaler_mean'],dtype=float)
    sd=np.asarray(checkpoint['scaler_scale'],dtype=float)
    co=np.asarray(checkpoint['coefficients'],dtype=float)
    bias=float(checkpoint['intercept'])
    if (x.ndim!=2 or x.shape[1]!=len(cols) or
         any(v.shape!=(len(cols),) for v in (mu,sd,co)) or
         not np.isfinite(x).all() or not np.isfinite([*mu,*sd,*co,bias]).all() or
         not np.all(sd>0)):
         raise ValueError('Invalid extracted synthetic classifier coefficients')
    return expit(((x-mu)/sd)@co + bias)


def score_native_tile(gray:np.ndarray, candidates:list[dict], *,photo:str,
                      checkpoints:dict[str,dict]) -> list[dict]:
    if not isinstance(candidates,list) or photo not in ('005','006'):
        raise ValueError('Bad native candidate provenance')
    source='006' if photo=='005' else '005'
    ids=[c['draft_id'] for c in candidates]
    if len(ids)!=len(set(ids)):
        raise ValueError('Ambiguous machine candidate identities')
    if not candidates:
        return []
    maps=ridge_evidence(gray)
    old=np.stack([path_features(gray,c['points_xy'],maps=maps) for c in candidates])
    extra=np.stack([texture_features(gray,c['points_xy']) for c in candidates])
    extended=merge_features(old,extra)
    a=_linear_predict(old,checkpoints['existing_S43'],source,photo,False)
    b=_linear_predict(extended,checkpoints['S43_plus_bead_texture_S53'],source,photo,True)
    if not (np.isfinite(a).all() and np.isfinite(b).all()
            and np.all((a>=0)&(a<=1)) and np.all((b>=0)&(b<=1))):
        raise ValueError('Invalid simulated renderer similarity scores')
    rows=[]
    for i, ident in enumerate(ids):
        rows.append({
            'draft_id':ident,
            'S43_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB':round(float(a[i]),6),
            'S53_simulated_bead_origin_score_NOT_NATIVE_PHYSICS_PROB':round(float(b[i]),6),
            'S53_minus_S43_score_difference_NOT_ACCURACY':round(float(b[i]-a[i]),6),
            'bead_texture_descriptor':dict(zip(NEW_NAMES,np.round(extra[i],5).tolist())),
            'source_image':photo,
            'classifier_train_photo':source,
            'native_physical_identity':'UNKNOWN',
            'native_track_confidence_AVAILABLE':False,
            'candidate_preserved':True,
            'physical_auto_join_authorized':False,
        })
    return rows
