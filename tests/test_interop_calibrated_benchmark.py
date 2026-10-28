"""Detector-specific TRAIN-only score fit and frozen CAL, source-only synthetic QA."""
import json
import numpy as np
import pytest
from scripts.research.bubble_chamber import interop_calibrated_benchmark as b


def test_seeds_and_source_partitions_do_not_overlap():
    assert set(b.SEEDS)=={'train','cal','test'}
    assert len(set(b.SEEDS.values()))==3
    assert b.SCENE_N=={'train':24,'cal':16,'test':24}
    assert len({t['id'] for p in b.SEEDS.values() for t in b.generated_templates(p)})==32*3
    assert b.CONDITIONS==('clean','blur_sigma_11','scratch_grain')


def test_score_fit_cannot_learn_from_all_negative_or_too_few_scenes():
    columns=len(b.FIELDS)
    sample=np.ones(columns)
    scene_data=[[ ([],[],np.tile(sample,(5,1)),[0]*5)] for i in range(8)]
    model,receipt=b.train_model(scene_data)
    assert model is None and receipt['trained'] is False
    assert np.array_equal(b.score_model(model)([dict((k,1.0) for k in b.FIELDS)]),np.array([0.]))
    # Enough independent scenes/classes, still only synthetic QA.
    data=[]
    for i in range(8):
        y=[1,0,0,0] if i%2==0 else [0,1,0,0]
        x=np.tile(sample,(4,1));x[0,0]=i*.4;x[1,1]=-i*.2
        data.append([(None,None,x,y)])
    fit,receipt=b.train_model(data)
    assert fit is not None and receipt['trained'] and receipt['positive']==8
    assert len(receipt['frozen_private_parameters_sha256'])==64


def test_cal_policy_abstains_if_wrong_links_breach_frozen_limit():
    base=dict(selected_synthetic_correct_links=0,synthetic_noncorrect_selected=5,
              complete_three_piece_paths_synthetic=0,shape_true_fragments_matched=2,
              solver_failure=0,candidate_polylines=9,candidate_pairs_after_strict_gate=14)
    observations={(s,t,c):[base.copy()] for s in b.STRATEGIES[1:] for t in b.CUTS for c in b.CONDITIONS}
    policies,winner=b.cal_policy(observations)
    assert winner=='none'
    assert all(v['threshold']==1.0 and not v['eligible'] for v in policies.values())
    good=dict(base,selected_synthetic_correct_links=5,synthetic_noncorrect_selected=1,
              complete_three_piece_paths_synthetic=2)
    for c in b.CONDITIONS:
        observations[('port_greedy',.90,c)]=[good]
    policies,winner=b.cal_policy(observations)
    assert winner=='port_greedy' and policies[winner]['threshold']==.9


def test_blind_feature_interface_does_not_accept_reference_geometry():
    from inspect import signature
    assert list(signature(b.extract_pair_features).parameters)==['gray','detector']
    assert list(signature(b.predict_one).parameters)==['lines','scorer','cutoff','strategy']
    assert b.PROTOCOL.is_file()


def test_smoke_is_explicitly_not_confirmatory_and_frozen_model_by_detector():
    sample=b.run(train_n=2,cal_n=2,test_n=2)
    assert sample['smoke_NOT_CONFIRMATORY'] is True
    assert sample['mode']=='GENERATED_ONLY_NOT_H50Z'
    assert not sample['H50Z_native_physics_improvement_demonstrated']
    assert not sample['physical_auto_joins_authorized']
    for name in sample['detectors']:
        assert sample['CAL_selected'][name] in b.STRATEGIES
        assert sample['TEST_aggregates'][name]['clean']['none']['selected_synthetic_correct_links']==0
        assert sample['TEST_aggregates'][name]['clean']['none']['synthetic_noncorrect_selected']==0
    with pytest.raises(ValueError):b.run(train_n=100,cal_n=2,test_n=2)
