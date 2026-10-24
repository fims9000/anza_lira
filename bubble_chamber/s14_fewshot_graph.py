"""Small-review Graph-LIRA pair classifier and cautious unattended join policy.

Graph inference needs **instance-level** (same/different) expert review, not
only foreground pixel scribbles. Unknown review decisions never train the
classifier. No future image requires humans for scoring: humans are needed
initially for labels, and again only for audited validation/exception handling.

Predictions do NOT become trusted physical tracks until the independent-run
validation gate is met. This module never claims real-film validation itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .real_sparse_lira import _features
from .graph_safety import select_global_links


@dataclass(frozen=True)
class LinkPolicy:
    max_validation_false_positive_fraction:float = .01
    global_margin:float = .08
    min_independent_validation_frames:int = 5
    require_independent_run:bool = True

    def validate(self):
        if not(0 <= self.max_validation_false_positive_fraction < .1 and
               0 <= self.global_margin <= .5 and self.min_independent_validation_frames>=2):
            raise ValueError('invalid Graph-LIRA policy')


def features_from_segment_pair(item:dict)->np.ndarray:
    """Features from visible geometry; no truth/labels enter this function."""
    left=np.asarray(item['left_points_xy'],dtype=np.float64)
    right=np.asarray(item['right_points_xy'],dtype=np.float64)
    if left.ndim!=2 or right.ndim!=2 or left.shape[1:]!=(2,) or right.shape[1:]!=(2,):
        raise ValueError('expected Hx2 candidate coordinates')
    if min(len(left),len(right))<3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('need >=3 finite points for both fragments')
    if np.linalg.norm(left[-1]-left[-3])<1e-6 or np.linalg.norm(right[2]-right[0])<1e-6:
        raise ValueError('degenerate end tangent')
    _,generic,axial=_features(left,right)
    return np.asarray(axial,dtype=np.float64)


def _validate_row(item:dict):
    for field in ['film_id','run_id','frame_id','scene_id','left_segment','right_segment','source_sha256']:
        if not str(item.get(field,'')).strip():raise ValueError(f'missing {field}')
    if item['left_segment']==item['right_segment']:raise ValueError('same segment cannot link itself')
    if len(str(item['source_sha256']))!=64:raise ValueError('expected original image sha256')
    if item.get('expert_label') not in {'same_track','different_track','uncertain',None}:
        raise ValueError('invalid expert review label')
    features_from_segment_pair(item)


def fit_fewshot(rows:list[dict], *, source_run_id:str, seed:int=20261024):
    """Trains only source_run human decisions; does not inspect target-run truth."""
    reviewed=[]
    for row in rows:
        _validate_row(row)
        if row['run_id']==source_run_id and row.get('expert_label') in {'same_track','different_track'}:
            reviewed.append(row)
    if len(reviewed)<10:
        raise ValueError('Need >=10 expert-reviewed pair decisions')
    y=np.asarray([int(r['expert_label']=='same_track') for r in reviewed],dtype=np.int64)
    if set(y)!={0,1}:
        raise ValueError('Need expert-confirmed same AND different relations')
    features=np.stack([features_from_segment_pair(r) for r in reviewed])
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=seed))
    model.fit(features,y)
    return model,{'human_reviewed_pairs':len(reviewed),'positive':int(y.sum()),'negative':int((1-y).sum()),
                  'source_run_id':source_run_id,'feature_source':'observed geometry only; NOT AZConv visual features',
                  'unreviewed_used_as_negatives':0}


def calibrate_and_score(model,calibration_rows:list[dict],unlabeled_candidates:list[dict],
                        *,train_run_id:str, policy:LinkPolicy=LinkPolicy(), require_operator_certification:bool=True,
                        external_physics_validation:dict|None=None):
    """Scores all candidates, calibrates only on *separate* labeled frames/runs.

    Returns a provisional graph for software validation. An actual unattended
    output requires independent-run validation plus an explicit operator gate.
    """
    policy.validate()
    if not calibration_rows or not unlabeled_candidates:raise ValueError('empty validation or inference candidates')
    for r in calibration_rows+unlabeled_candidates:_validate_row(r)
    if any(r['expert_label'] not in {'same_track','different_track'} for r in calibration_rows):
        raise ValueError('calibration requires full human decisions on all supplied candidates')
    if any(r.get('expert_label') is not None for r in unlabeled_candidates):
        raise ValueError('inference candidates must not contain expert labels')
    val_runs={r['run_id'] for r in calibration_rows}
    unseen_run=(train_run_id not in val_runs)
    frame_count=len({(r['run_id'],r['frame_id']) for r in calibration_rows})
    features=np.stack([features_from_segment_pair(r) for r in calibration_rows])
    scores=model.predict_proba(features)[:,1]
    y=np.asarray([r['expert_label']=='same_track' for r in calibration_rows],bool)
    if not (~y).any() or not y.any():raise ValueError('independent calibration needs both relation classes')
    neg=scores[~y]
    # Maximize recall at conservative candidate FPR. Threshold >max if no safe
    # positive threshold exists; validation may cause fully abstaining graph.
    thresholds=np.r_[np.nextafter(float(max(scores)),np.inf),np.unique(scores)]
    valid=[t for t in thresholds if float((neg>=t).mean()) <=policy.max_validation_false_positive_fraction]
    threshold=float(min(valid))
    probs=model.predict_proba(np.stack([features_from_segment_pair(r) for r in unlabeled_candidates]))[:,1]
    struct=[]
    for row in unlabeled_candidates:
        struct.append({'scene_id':row['scene_id'],
                       'pair':[int(row['left_segment']),'B',int(row['right_segment']),'A']})
    proposed,reason=select_global_links(struct,probs,threshold=threshold,margin=policy.global_margin)
    # A separate run is a necessary, NOT sufficient, criterion for deployment.
    safe_to_release=(not policy.require_independent_run or unseen_run) and frame_count>=policy.min_independent_validation_frames
    # A second independent REAL test run, externally verified photo-track masks
    # and calibration are required for unattended PHYSICAL interpretation.
    # These are not available in S14; same-run synthetic reviews cannot pass.
    evidence=external_physics_validation or {}
    third_run=evidence.get('heldout_real_test_run_id')
    valid_external=bool(
        isinstance(third_run,str) and third_run and third_run!=train_run_id and third_run not in val_runs
        and int(evidence.get('heldout_reviewed_frames',0))>=policy.min_independent_validation_frames
        and evidence.get('real_photo_instance_masks_expert_confirmed') is True
        and isinstance(evidence.get('stereo_calibration_id'),str) and evidence['stereo_calibration_id']
        and isinstance(evidence.get('field_map_version'),str) and evidence['field_map_version']
        and isinstance(evidence.get('filling_medium'),str) and evidence['filling_medium']
        and isinstance(evidence.get('review_audit_sha256'),str) and len(evidence['review_audit_sha256'])==64
        and evidence.get('heldout_false_join_rate_per_candidate') is not None
        and 0<=float(evidence['heldout_false_join_rate_per_candidate'])<=policy.max_validation_false_positive_fraction
    )
    unattended=bool(safe_to_release and valid_external and not require_operator_certification)
    return {'schema':'s14_provisional_graph_lira_inference_v1',
            'source_run_id':train_run_id,'calibration_run_ids':sorted(val_runs),
            'calibration_frames':frame_count,'different_run_calibration':unseen_run,
            'threshold_fitted_on_validation':threshold,
            'validation_positives':int(y.sum()),'validation_negatives':int((~y).sum()),
            'validation_candidate_false_positive_rate':float(((scores>=threshold)&~y).sum()/max((~y).sum(),1)),
            'links':[{'scene_id':row['scene_id'],'left_segment':row['left_segment'],
                      'right_segment':row['right_segment'],
                      'probability_not_calibrated_certainty':round(float(p),5),
                      'proposed':bool(chosen),
                      'output_status':'PROVISIONAL_REVIEW_REQUIRED' if bool(chosen) and not unattended else
                                      ('AUTOMATED_ONLY_AFTER_EXTERNAL_VALIDATION' if chosen else 'REJECTED_OR_UNCERTAIN'),
                      'reason':why} for row,p,chosen,why in zip(unlabeled_candidates,probs,proposed,reason)],
            'unattended_release_allowed':unattended,
            'independent_physics_validation_received':valid_external,
            'warning':'Unattended release flag is NOT proof of physical accuracy; this code cannot verify camera/B-field calibration, new filling conditions, or real mask quality.'}


def pair_rows_from_review(pilot_result:dict,expert_decisions:dict,*,film_id:str,run_id:str,frame_id:str)->list[dict]:
    """Turn P0 fragment-review decisions into REAL training instances.

    All unreviewed candidates stay UNLABELED, not inferred negatives. Source
    SHA and proposal identity are revalidated against the exact P0 result.
    """
    from .review import pair_key,validate_labels
    if not all(isinstance(x,str) and x.strip() for x in (film_id,run_id,frame_id)):
        raise ValueError('film/run/frame identities are mandatory')
    validate_labels(pilot_result,expert_decisions)
    segments={int(s['segment_id']):s for s in pilot_result['segments']}
    human={x['pair_key']:x['label'] for x in expert_decisions['labels']}
    accepted=[]
    for pair in pilot_result['join_proposals']:
        key=pair_key(pair)
        i,j=int(pair['segment_A']),int(pair['segment_B'])
        a=np.asarray(segments[i]['measurement_xy_px'],dtype=float)
        b=np.asarray(segments[j]['measurement_xy_px'],dtype=float)
        # Orient A to finish at the considered endpoint, and B to begin at it.
        if pair['endpoint_A']=='A':a=a[::-1].copy()
        if pair['endpoint_B']=='B':b=b[::-1].copy()
        row={'film_id':film_id,'run_id':run_id,'frame_id':frame_id,
             'scene_id':f'{run_id}:{frame_id}',
             'left_segment':i,'right_segment':j,
             'left_points_xy':a.tolist(),'right_points_xy':b.tolist(),
             'source_sha256':pilot_result['source_sha256_grayscale_pixels'],
             'expert_label':human.get(key), 'review_pair_key':key,
             'source_kind':'human-reviewed original photographic fragments'}
        _validate_row(row)
        accepted.append(row)
    return accepted


def json_model_parameters(model)->dict:
    """Read-only portable linear classifier; no pickle required at inference."""
    scaler=model.named_steps['standardscaler']
    logistic=model.named_steps['logisticregression']
    if logistic.coef_.shape!=(1,scaler.mean_.shape[0]):
        raise ValueError('not a binary graph relation model')
    return {'schema':'s14_graph_lira_linear_predictor_v1',
            'feature_names':'geometry_axial_fuzzy (14 fixed observed features)',
            'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':logistic.coef_[0].tolist(),'intercept':float(logistic.intercept_[0]),
            'threshold_is_run_specific':'requires separate reviewed validation',
            'never_treat_unlabeled_as_negative':True}


def portable_predict(rows:list[dict],parameters:dict)->np.ndarray:
    if parameters.get('schema')!='s14_graph_lira_linear_predictor_v1':
        raise ValueError('unsupported predictor export')
    xx=np.stack([features_from_segment_pair(r) for r in rows])
    mean=np.asarray(parameters['mean']);scale=np.asarray(parameters['scale'])
    coef=np.asarray(parameters['coef'])
    if xx.shape[1]!=len(coef) or not np.all(scale>0):raise ValueError('feature mismatch')
    z=np.clip(((xx-mean)/scale)@coef + float(parameters['intercept']),-40,40)
    return 1/(1+np.exp(-z))
