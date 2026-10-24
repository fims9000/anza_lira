import numpy as np
import pytest
from bubble_chamber.s14_fewshot_graph import features_from_segment_pair,fit_fewshot,calibrate_and_score,LinkPolicy


def row(i,run='r1',label='different_track'):
    x=np.arange(6,dtype=float)
    left=np.column_stack([x, .05*x*x])
    right=np.column_stack([x+7+i%3, .05*(x+7+i%3)**2 + (0 if label=='same_track' else 4)])
    return {'film_id':run,'run_id':run,'frame_id':f'{run}_f{i//4}',
            'scene_id':f'{run}_f{i//4}','left_segment':int(2*i+1),'right_segment':int(2*i+2),
            'left_points_xy':left.tolist(),'right_points_xy':right.tolist(),
            'source_sha256':'0'*64,'expert_label':label}


def training_rows():
    return [row(i,label='same_track' if i%2==0 else 'different_track') for i in range(26)]


def test_features_never_consume_labels():
    a=row(2,label='same_track');b=dict(a,expert_label='different_track')
    assert np.array_equal(features_from_segment_pair(a),features_from_segment_pair(b))


def test_rejects_unlabeled_or_single_class_training():
    with pytest.raises(ValueError):fit_fewshot([row(i,label='uncertain') for i in range(16)],source_run_id='r1')
    with pytest.raises(ValueError):fit_fewshot([row(i,label='same_track') for i in range(16)],source_run_id='r1')


def test_fewshot_train_and_provisional_inference():
    model,stats=fit_fewshot(training_rows(),source_run_id='r1')
    assert stats['human_reviewed_pairs']==26
    assert stats['unreviewed_used_as_negatives']==0
    val=[row(i,'r2','same_track' if i%2==0 else 'different_track') for i in range(26)]
    unknown=[row(i,'r3','different_track') for i in range(5)]
    for x in unknown:x['expert_label']=None
    result=calibrate_and_score(model,val,unknown,train_run_id='r1')
    assert len(result['links'])==5
    assert all(x['output_status']!='AUTOMATED_ONLY_AFTER_EXTERNAL_VALIDATION' for x in result['links'])
    assert result['unattended_release_allowed'] is False
    assert result['different_run_calibration'] is True


def test_same_run_never_releases_without_cross_run():
    model,_=fit_fewshot(training_rows(),source_run_id='r1')
    val=[row(i,'r1','same_track' if i%2==0 else 'different_track') for i in range(26)]
    unknown=[row(i,'r2','different_track') for i in range(5)]
    for x in unknown:x['expert_label']=None
    result=calibrate_and_score(model,val,unknown,train_run_id='r1',require_operator_certification=False)
    assert result['unattended_release_allowed'] is False


def test_inference_labels_or_invalid_identity_rejected():
    model,_=fit_fewshot(training_rows(),source_run_id='r1')
    val=[row(i,'r2','same_track' if i%2==0 else 'different_track') for i in range(26)]
    with pytest.raises(ValueError,match='must not contain'):
        calibrate_and_score(model,val,[row(1,'r3','same_track')],train_run_id='r1')
    corrupt=row(7); corrupt['left_points_xy']=[[0,0],[1,1]]
    with pytest.raises(ValueError):features_from_segment_pair(corrupt)


def test_portable_model_prediction_matches_sklearn():
    from bubble_chamber.s14_fewshot_graph import json_model_parameters,portable_predict
    model,_=fit_fewshot(training_rows(),source_run_id='r1')
    points=[row(i,run='r2',label=None) for i in range(9)]
    actual=model.predict_proba(np.stack([features_from_segment_pair(x) for x in points]))[:,1]
    predicted=portable_predict(points,json_model_parameters(model))
    assert np.allclose(actual,predicted,atol=1e-7)


def test_real_review_integration_preserves_unknown_labels():
    from bubble_chamber.review import result_digest,pair_key,SCHEMA
    from bubble_chamber.s14_fewshot_graph import pair_rows_from_review
    result={'source_sha256_grayscale_pixels':'1'*64,'pipeline':'p0_test',
      'segments':[{'segment_id':1,'measurement_xy_px':[[i,0] for i in range(6)]},
                  {'segment_id':2,'measurement_xy_px':[[i+8,1] for i in range(6)]},
                  {'segment_id':3,'measurement_xy_px':[[i+8,6] for i in range(6)]}],
      'join_proposals':[{'segment_A':1,'endpoint_A':'B','segment_B':2,'endpoint_B':'A'},
                        {'segment_A':1,'endpoint_A':'B','segment_B':3,'endpoint_B':'B'}]}
    review={'schema':SCHEMA,'source_sha256_grayscale_pixels':result['source_sha256_grayscale_pixels'],
           'pipeline':result['pipeline'],'result_sha256':result_digest(result),
           'labels':[{'pair_key':pair_key(result['join_proposals'][0]),'label':'same_track','comment':''}]}
    rows=pair_rows_from_review(result,review,film_id='filmH',run_id='h2',frame_id='frame01')
    assert len(rows)==2
    assert rows[0]['expert_label']=='same_track' and rows[1]['expert_label'] is None
    assert rows[1]['right_points_xy'][0]==[13.,6.]
    with pytest.raises(ValueError,match='SHA mismatch'):
        pair_rows_from_review(result,{**review,'source_sha256_grayscale_pixels':'2'*64},film_id='filmH',run_id='h2',frame_id='frame01')


def test_other_run_validation_alone_not_enough_for_physical_auto_joins():
    model,_=fit_fewshot(training_rows(),source_run_id='r1')
    val=[row(i,'r2','same_track' if i%2==0 else 'different_track') for i in range(26)]
    unknown=[row(i,'r3',None) for i in range(5)]
    result=calibrate_and_score(model,val,unknown,train_run_id='r1',require_operator_certification=False)
    assert result['unattended_release_allowed'] is False
    assert not result['independent_physics_validation_received']
