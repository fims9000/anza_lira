"""S77 public numerical receipts: synthetic-only, no native expert GT."""
from hashlib import sha256
import json
from pathlib import Path
from bubble_chamber.s77_short_gap_instances import ARMS
from scripts.research.bubble_chamber import s77_calibrate_then_test_short_gaps as task

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'
CAL=DOC/'S77_PUBLIC_FROZEN_CAL_2026_10_26.json'
TEST=DOC/'S77_PUBLIC_FRESH_TEST_2026_10_26.json'
NATIVE=DOC/'S77_PUBLIC_NATIVE_UNLABELED_SMOKE_2026_10_26.json'


def test_public_calibration_was_frozen_before_new_test():
    cal=json.loads(CAL.read_text())
    test=json.loads(TEST.read_text())
    assert cal['experiment_id']==task.EXPERIMENT_ID==test['experiment_id']
    assert cal['phase']=='cal' and test['phase']=='test'
    assert cal['phase_scene_count_per_seed']==16
    assert test['phase_scene_count_per_seed']==24
    assert test['frozen_S77_CAL_SHA256']==sha256(CAL.read_bytes()).hexdigest()
    assert test['protocol_sha256']==cal['protocol_sha256']==task.digest(task.PROTOCOL)
    assert test['operator_sha256']==cal['operator_sha256']==task.digest(task.OPERATOR)
    assert test['source_CAL_sha256']==task.LOCKED_S72_CAL_SHA256
    assert set(cal['directions'])==set(test['directions'])=={'005_to_006','006_to_005'}
    assert all(x['CAL_decision']['CAL_selected']=='baseline' for x in cal['directions'].values())
    assert all(x['CAL_selected']=='baseline' for x in test['directions'].values())
    assert not cal['physical_joins_authorized'] and not test['physical_joins_authorized']
    assert test['native_GT_track_count']==cal['native_GT_track_count']==0
    assert not test['smoke_NON_CONFIRMATORY']


def test_public_stats_are_replay_stable_and_preserve_original_default():
    data=json.loads(TEST.read_text())
    exp={
      'clean':{'baseline':(54,33,14),'short_gap_strict':(47,31,12),'short_gap_relaxed':(37,21,8)},
      'blur_sigma_11':{'baseline':(15,24,4),'short_gap_strict':(12,21,3),'short_gap_relaxed':(10,18,2)},
      'blur_sigma_16':{'baseline':(11,18,1),'short_gap_strict':(11,16,1),'short_gap_relaxed':(8,14,1)},
    }
    for cond,arms in exp.items():
        assert set(data['all_TEST_aggregates'][cond])==set(ARMS)
        for arm,values in arms.items():
            x=data['all_TEST_aggregates'][cond][arm]
            assert (x['selected_synthetic_correct_links'],x['synthetic_noncorrect_selected'],
                    x['complete_three_piece_paths_synthetic'])==values
            assert x['scenes']==96
            assert x['selected_synthetic_correct_links']+x['synthetic_noncorrect_selected']==x['selected_provisional_links']
            assert x['shape_true_both_endpoints_close']+x['shape_true_one_or_both_endpoints_far']==x['shape_true_fragments_matched']
            assert x['solver_failure']==0
        assert data['CAL_selected_TEST_aggregate'][cond]==data['all_TEST_aggregates'][cond]['baseline']


def test_private_native_photo_has_zero_gt_or_auto_joins():
    d=json.loads(NATIVE.read_text())
    assert d['input_native_unlabelled_crops']==39
    assert d['native_physics_GT_count']==0
    assert d['physical_auto_join_authorized'] is False
    assert d['graph_proposals_are_unknown'] is True
    assert d['aggregates']['baseline']['graph_advisory_joins_UNKNOWN']==3
    assert d['aggregates']['short_gap_strict']['pixel_supported_short_gap_repairs']==3
    assert d['aggregates']['short_gap_relaxed']['pixel_supported_short_gap_repairs']==4
    assert all(x['native_physics_GT_count']==0 for x in d['aggregates'].values())
