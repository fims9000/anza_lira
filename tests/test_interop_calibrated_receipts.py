"""Immutable source-only synthetic CAL/TEST receipts; never H50Z evidence."""
from hashlib import sha256
import json
from pathlib import Path
from scripts.research.bubble_chamber import interop_calibrated_benchmark as b

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/bubble_chamber'


def data():return json.loads((DOCS/'S85_INTEROP_CALIBRATED_GENERATED_TOY_RESULTS_2026_10_28.json').read_text())


def test_frozen_source_and_full_phase_receipt_did_not_use_test_to_choose():
    j=data()
    assert j['schema']==b.SCHEMA
    assert j['experiment_id']==b.ID
    assert j['mode']=='GENERATED_ONLY_NOT_H50Z'
    assert not j['smoke_NOT_CONFIRMATORY']
    assert j['frozen_design_budgets']==b.SCENE_N
    assert (j['train_scenes'],j['cal_scenes'],j['test_scenes'])==(24,16,24)
    assert not j['H50Z_native_physics_improvement_demonstrated']
    assert j['physical_GT_count']==0 and not j['physical_auto_joins_authorized']
    assert j['cal_frozen_before_test'] is True
    for path,digest in j['source_locks'].items():
        assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path


def test_all_three_detectors_train_on_own_pairs_but_cal_abstains():
    j=data()
    expected={'hessian-ridge-v1':(83,245,20),
              'opencv-lsd-v1':(109,842,20),
              'opencv-hough-p-v1':(23,649,11)}
    assert set(j['detectors'])==set(expected)
    for d,counts in expected.items():
        tr=j['models'][d]
        assert tr['trained']
        assert (tr['positive'],tr['negative'],tr['independent_positive_source_scenes'])==counts
        assert len(tr['frozen_private_parameters_sha256'])==64
        assert j['CAL_selected'][d]=='none'
        for name in ('node_mutual','port_greedy','global_milp'):
            assert j['CAL_strategies'][d][name]['eligible'] is False
            assert j['CAL_strategies'][d][name]['threshold']==1.
        for c in b.CONDITIONS:
            for strategy,row in j['TEST_aggregates'][d][c].items():
                assert row['scenes']==24 and row['selected_synthetic_correct_links']==0
                assert row['synthetic_noncorrect_selected']==0
                assert row['complete_three_piece_paths_synthetic']==0
                assert row['solver_failure']==0


def test_public_payload_never_contains_pixels_measured_points_or_weights():
    raw=(DOCS/'S85_INTEROP_CALIBRATED_GENERATED_TOY_RESULTS_2026_10_28.json').read_text()
    for bad in ('source_image_base64','image_pixel_array','model_coef','scaler_mean','points_xy',
                'real_H50Z_GT_confirmed', 'model_checkpoint_base64'):
        assert bad not in raw


def test_registry_identifies_negative_result_without_physics_claims():
    registry=json.loads((DOCS/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    matches=[r for r in registry['experiments'] if r.get('id')==b.ID]
    assert len(matches)==1
    r=matches[0]
    assert r['native_GT_count']==0 and r['real_H50Z_test_executed'] is False
    assert not r['physical_auto_join_authorized']
    assert r['CAL_selected_all_detectors']=='none'
    for key in ('protocol','benchmark','result','report','coords','verified_runtime','readme'):
        assert sha256((ROOT/r[key]).read_bytes()).hexdigest()==r[key+'_sha256']
