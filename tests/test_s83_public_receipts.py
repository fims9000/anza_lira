import json
from hashlib import sha256
from pathlib import Path
from scripts.research.bubble_chamber import s83_calibrate_non_saturating as s83
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'

def read(name):return json.loads((DOC/name).read_text())

def test_immutable_test_consumes_frozen_cal_and_non_saturating_quantiles():
    p=DOC/'S83_PUBLIC_FROZEN_CAL_2026_10_26.json';cal=json.loads(p.read_text())
    test=read('S83_PUBLIC_FRESH_TEST_2026_10_26.json')
    assert test['frozen_CAL_sha256']==sha256(p.read_bytes()).hexdigest()
    assert cal['protocol_sha256']==test['protocol_sha256']==s83.digest(s83.PROTOCOL)
    assert cal['operator_sha256']==test['operator_sha256']==s83.digest(s83.OPERATOR)
    assert cal['S82_source_resampling_sha256']==test['S82_source_resampling_sha256']==s83.digest(s83.S82_RESAMPLING_CODE)
    assert all(x['CAL_decision']['CAL_selected']=='baseline' for x in cal['directions'].values())
    for row in cal['directions'].values():
        q=row['CAL_only_local_thresholds']
        assert 0<q['s83_local_q40']<q['s83_local_q70']<1.
    assert test['physical_deployment_authorized'] is False


def test_new_TEST_clean_blur_metrics_lock():
    j=read('S83_PUBLIC_FRESH_TEST_2026_10_26.json')['all_TEST_aggregates']
    expected={'clean':[(46,36,14),(50,41,14),(48,39,14)],
              'blur_sigma_11':[(17,21,3),(17,21,3),(17,21,3)],
              'blur_sigma_16':[(11,14,2),(11,15,2),(11,15,2)]}
    for condition,rows in expected.items():
        assert [(j[condition][a]['selected_synthetic_correct_links'],
                j[condition][a]['synthetic_noncorrect_selected'],
                j[condition][a]['complete_three_piece_paths_synthetic']) for a in s83.ARMS]==rows
        assert all(j[condition][a]['scenes']==96 for a in s83.ARMS)


def test_native_physical_truth_not_claimed():
    j=read('S83_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json')
    assert j['tiles']==39 and j['native_physics_GT_count']==0
    assert j['physical_auto_joins_authorized'] is False and j['all_machine_joins']=='UNKNOWN'
    assert [(j['arms'][a]['candidate_lines'],j['arms'][a]['graph_joins_advisory_UNKNOWN']) for a in s83.ARMS]==[(83,3),(90,4),(85,3)]


def test_registry_sha_provenance_and_no_private_data():
    reg=read('EXPERIMENT_IDS_MACHINE_ONLY.json')
    records=[x for x in reg['experiments'] if x.get('id')==s83.EXPERIMENT_ID]
    assert len(records)==1
    item=records[0]
    for name in ('protocol','predictor','benchmark','native_audit','calibration','test','native_smoke','report'):
        assert item[name+'_sha256']==sha256((ROOT/item[name]).read_bytes()).hexdigest()
    assert item['native_particle_GT_count']==0 and item['physical_auto_join_authorized'] is False
    for x in ('S83_PUBLIC_FROZEN_CAL_2026_10_26.json',
              'S83_PUBLIC_FRESH_TEST_2026_10_26.json','S83_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json'):
        raw=(DOC/x).read_text()
        assert all(t not in raw for t in ('points_xy','data:image','_mined00.png','raw_image_base64'))
