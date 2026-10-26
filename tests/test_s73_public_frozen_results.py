"""Public S73 result cannot silently imply physics GT or alter S72 choice."""
from hashlib import sha256
import json
from pathlib import Path
from scripts.research.bubble_chamber import s73_stress_test_pixel_graph as s73

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'docs/bubble_chamber/S73_PUBLIC_FRESH_DOMAIN_STRESS_2026_10_26.json'


def test_stress_result_matches_frozen_protocol_without_native_photo_data():
    j=json.loads(RESULT.read_text())
    assert j['experiment_id']=='ANZA-LIRA-BC-S73-20261026'
    assert j['phase']=='fresh_stress_TEST'
    assert j['S73_protocol_sha256']==sha256(s73.PROTOCOL.read_bytes()).hexdigest()
    assert j['S72_CAL_sha256']==s73.FROZEN_S72_CAL_SHA256
    assert j['scenes_per_seed']==24
    assert j['conditions']==list(s73.CONDITIONS)
    assert j['native_H50Z_verified_GT_count']==0
    assert j['native_physical_auto_joins_authorized'] is False
    assert j['frozen_method_stress_gate']['conditional_synthetic_only_passed'] is False
    assert j['frozen_method_stress_gate']['native_physics_promotion_authorized'] is False
    assert j['frozen_method_stress_gate']['evaluated_direction_stressor_cells']==8
    assert j['directions']['005_to_006']['frozen_CAL_selected']=='s68_branch_endpoint_refine'
    assert j['directions']['006_to_005']['frozen_CAL_selected']=='s39_original'
    assert sum(len(d['test_seeds']) for d in j['directions'].values())==4


def test_stress_results_have_exhaustive_noncorrect_accounting():
    j=json.loads(RESULT.read_text())
    for kind in j['conditions']:
        d=j['overall_condition_totals'][kind]
        for row in [*d['arm_totals'].values(),d['frozen_S72_direction_choice']]:
            assert row['scene_count']==96
            assert row['selected_provisional_links']==(row['selected_synthetic_correct_links']+
                  row['synthetic_noncorrect_selected'])
            assert 0<=row['complete_three_piece_paths_synthetic']<=192
            assert row['solver_failure']==0
    totals={name:j['overall_condition_totals'][name]['frozen_S72_direction_choice'] for name in j['conditions']}
    assert [(totals[k]['selected_synthetic_correct_links'],totals[k]['synthetic_noncorrect_selected'],
             totals[k]['complete_three_piece_paths_synthetic']) for k in j['conditions']]==[
        (50,35,14),(50,38,13),(13,20,2),(43,36,7),(35,45,5)]


def test_public_aggregate_contains_no_native_pixels_or_raw_frame_ids():
    src=RESULT.read_text()
    for forbidden in ('/mnt/data/','data:image/','base64,','H50Z_030_01_','_PRIVATE_source_frames',
                      'points_xy','polyline_xy','original_photograph','source_frame_name'):
        assert forbidden not in src
