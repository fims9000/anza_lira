"""Public S74 negative result, exact CAL and native truth boundary."""
from pathlib import Path
from hashlib import sha256
import json
from scripts.research.bubble_chamber import s74_calibrate_then_test_unsharp as s74

ROOT=Path(__file__).resolve().parents[1]/'docs/bubble_chamber'
CAL=ROOT/'S74_PUBLIC_FROZEN_CAL_2026_10_26.json'
TEST=ROOT/'S74_PUBLIC_FRESH_TEST_2026_10_26.json'


def test_immutable_CAL_selection_and_nonphysical_output():
    a=json.loads(CAL.read_text());b=json.loads(TEST.read_text())
    assert a['phase']=='cal' and b['phase']=='test'
    assert b['frozen_S74_CAL_sha256']==sha256(CAL.read_bytes()).hexdigest()
    assert a['experiment_id']==b['experiment_id']=='ANZA-LIRA-BC-S74-20261026'
    assert b['S74_protocol_sha256']==sha256(s74.PROTOCOL.read_bytes()).hexdigest()
    assert not b['native_physical_auto_join_authorized']
    assert b['native_H50Z_verified_GT_count']==0
    assert b['H50Z_backgrounds_one_physical_run'] is True
    assert {d['frozen_CAL_unsharp_choice'] for d in b['directions'].values()}=={'identity'}


def test_scientific_aggregates_and_failed_recall_risk_gate():
    b=json.loads(TEST.read_text());c=b['all_TEST_conditions']
    assert c['clean']['identity']['scene_count']==c['blur_sigma_11']['identity']['scene_count']==96
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected','complete_three_piece_paths_synthetic')
    for condition,expected in {
        'clean':{'identity':(59,37,17),'unsharp_065':(69,37,18),'unsharp_130':(73,36,17)},
        'blur_sigma_11':{'identity':(20,32,5),'unsharp_065':(20,44,5),'unsharp_130':(31,41,8)},
    }.items():
        for arm,values in expected.items():
            assert tuple(c[condition][arm][key] for key in keys)==values
            assert c[condition][arm]['selected_provisional_links']==(
                c[condition][arm]['selected_synthetic_correct_links']+
                c[condition][arm]['synthetic_noncorrect_selected'])
            assert c[condition][arm]['solver_failure']==0


def test_result_privacy_boundary():
    for path in (CAL,TEST):
        content=path.read_text()
        assert all(x not in content for x in ('/mnt/data/','data:image/','base64,',
            'H50Z_030_01_','_PRIVATE_source_frames','points_xy','polyline_xy'))
