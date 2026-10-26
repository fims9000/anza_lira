from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, ModuleType
import numpy as np
import cv2
import pytest

from bubble_chamber.s85_fragment_ranker import ARMS, FEATURE_NAMES, fit_train_only, model_sha256
from bubble_chamber.s85_image_disturbances import condition_pixels
from bubble_chamber.s85_selection_gate import choose_cal_only
from bubble_chamber.s85_train_reference_labels import label_extras_after_inference
from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85


def curve(y, x0=25, x1=155):
    return np.column_stack([np.linspace(x0,x1,39),np.full(39,y)]).astype(float)


def source_image(ink=20):
    a=np.full((192,192),205,np.uint8)
    cv2.line(a,(25,82),(155,82),ink,2)
    return a


def test_scratch_grain_is_deterministic_and_never_mutates_photo():
    im=source_image(); before=im.copy()
    a=condition_pixels(im,'scratch_grain',noise_seed=123)
    b=condition_pixels(im,'scratch_grain',noise_seed=123)
    assert a.shape==(192,192) and a.dtype==np.uint8 and np.array_equal(a,b)
    assert not np.array_equal(a,im) and np.array_equal(im,before)
    assert not np.array_equal(a,condition_pixels(im,'scratch_grain',noise_seed=124))
    assert np.array_equal(condition_pixels(im,'clean',noise_seed=1),im)
    with pytest.raises(ValueError):condition_pixels(im,'other',noise_seed=123)
    with pytest.raises(TypeError):condition_pixels(im,'scratch_grain',noise_seed=1,synthetic_ground_truth=3)


def test_ground_truth_only_exposed_in_evaluator_after_blind_features(monkeypatch):
    # Import the old evaluator on demand, with a stub for local source-only QA.
    package=ModuleType('bubble_chamber.s76_shape_endpoint_diagnostics')
    package.shape_only_match=lambda lines,refs: ({1:1,2:6},{},{})
    monkeypatch.setitem(sys.modules,'bubble_chamber.s76_shape_endpoint_diagnostics',package)
    reference=[curve(30+i*16) for i in range(7)]
    predicted=[curve(30),curve(46),curve(126),curve(182),curve(64)]
    assert label_extras_after_inference(predicted,1,reference)=={0:1,1:0,2:0,3:None}
    with pytest.raises(TypeError):s85.extract_blind_features(source_image(),[],[],[],None,
                                                       lambda *args:None,ground_truth=reference)
    with pytest.raises(ValueError):label_extras_after_inference(predicted,1,reference[:6])


def test_cal_selection_frozen_learned_requires_error_nonincrease():
    base=dict(selected_synthetic_correct_links=3,synthetic_noncorrect_selected=4,
              complete_three_piece_paths_synthetic=2,shape_true_fragments_matched=22,
              extra_accepted=0,solver_failure=0)
    grid={arm:{cond:dict(base) for cond in ('clean','blur_sigma_11','scratch_grain')}
          for arm in ARMS}
    grid['s85_learned_p95']['blur_sigma_11'].update(
        selected_synthetic_correct_links=5,shape_true_fragments_matched=24,
        synthetic_noncorrect_selected=5)
    assert choose_cal_only(grid,True)['CAL_selected']=='baseline'
    grid['s85_learned_p95']['blur_sigma_11']['synthetic_noncorrect_selected']=4
    assert choose_cal_only(grid,True)['CAL_selected']=='s85_learned_p95'
    assert choose_cal_only(grid,False)['CAL_selected']=='baseline'
    grid['s85_learned_p95']['clean']['synthetic_noncorrect_selected']=5
    assert choose_cal_only(grid,True)['CAL_selected']=='baseline'


def test_training_fails_closed_on_sparse_pixel_labelled_candidates():
    fake_img=source_image()
    source=curve(82);extra=curve(126)
    class FakeS80:
        @staticmethod
        def infer_from_pixels(*args):
            old=dict(polylines=[source],chosen=[],original_polylines_retained=1)
            return {'baseline':old,'s80_consensus_add6':{**old,'polylines':[source,extra]}}
    FakeScene=lambda *args,**kwargs:SimpleNamespace(gray=fake_img,nodes=['only-synthetic-TRAIN'],)
    model,counts=s85.train_source_model([Path('/PRIVATE/scene.png')],[1,2],101,4,
                       lambda x:np.ones(len(x)),.8,'s39_original',
                       ('test',),FakeScene,FakeS80,lambda *args:{0:[]},
                       lambda lines,nbase,ref:{0:None})
    assert not model['trained'] and model['physical_auto_joins_authorized'] is False
    assert counts['all_extra_proposals']==12 and counts['abstain']==12
    assert counts['positive']==counts['negative']==0


def test_inference_preserves_line_objects_and_no_model_will_join(monkeypatch):
    source=curve(82);extra=curve(110)
    baseline=dict(polylines=[source],chosen=[],pair_count=0,solver_success=True,
                  physics_auto_joins_authorized=False,original_polylines_retained=1,
                  added_pixel_instances=0,duplicate_suppressed=0,scale_consensus_rejected=0)
    cons=dict(baseline,polylines=[source,extra],added_pixel_instances=1)
    old=SimpleNamespace(infer_from_pixels=lambda *args:{'baseline':baseline,'s80_consensus_add6':cons})
    solves=SimpleNamespace(_solve=lambda *args: ([],0,True))
    dummy=(None,None,lambda *args:{0:[]},None,None,None,None,old,solves)
    monkeypatch.setattr(s85,'_deps',lambda:dummy)
    frozen={'schema':'S85_TRAIN_ONLY_REGULARIZED_LOGISTIC_V1','trained':False,
            'feature_names':list(FEATURE_NAMES)}
    result=s85.predict_blind(source_image(),lambda *args:[],.6,'s39_original',frozen)
    assert set(result)==set(ARMS)
    assert all(result[arm]['polylines'][0] is source for arm in ARMS)
    assert all(not result[arm]['chosen'] and not result[arm]['physics_auto_joins_authorized'] for arm in ARMS)
    assert result['s85_learned_p85']['added_pixel_instances']==0
    with pytest.raises(TypeError):s85.predict_blind(source_image(),None,.6,'s39_original',frozen,source_nodes=[])


def test_lock_checks_reject_train_model_drift_even_without_private_data(tmp_path,monkeypatch):
    model=fit_train_only(np.ones((2,len(FEATURE_NAMES))),[0,1],[0,1])
    base={'schema':s85.SCHEMA,'phase':'train','smoke_NOT_CONFIRMATORY':False,
          'locks':{'source':'frozen'},'source_split_sha256':'source-train-cal-test',
          's72_source_code_locks':{'S72':'locked'},
          'models':{x:{'model':model,'model_sha256':model_sha256(model)}
                    for x in ('005_to_006','006_to_005')}}
    ctx={'locks':base['locks'],'source_split_sha256':base['source_split_sha256'],
         's72_source_code_locks':base['s72_source_code_locks']}
    assert s85._model_receipt(base,ctx)
    base['models']['005_to_006']['model']['reason']='tampered'
    with pytest.raises(ValueError):s85._model_receipt(base,ctx)
    with pytest.raises(FileNotFoundError,match='Private H50Z'):
        s85.run('train',tmp_path/'MISSING_VISUALMETRO',tmp_path/'MISSING_TILES')


def test_source_only_seed_splits_are_unique_and_ready_without_faking_test():
    train=set(s85.TRAIN_SEEDS.values())
    cal=set(s85.CAL_SEEDS.values())
    test=set([v for seeds in s85.TEST_SEEDS.values() for v in seeds])
    old={20268401,20268411,20268403,20268404,20268413,20268414}
    assert not(train&cal or train&test or cal&test or old&test)
    assert s85.N_TRAIN==64 and s85.N_CAL==s85.N_TEST==24
    assert 'scratch_grain' in s85.TEST_CONDITIONS
    assert 'test' not in s85._deps.__code__.co_varnames  # no synthetic reference argument to _deps
