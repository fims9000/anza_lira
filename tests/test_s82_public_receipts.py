import json
from pathlib import Path
from hashlib import sha256
from scripts.research.bubble_chamber import s82_calibrate_per_line_then_test as s82
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'

def load(p):return json.loads((DOC/p).read_text())

def test_CAL_provenance_and_unchanged_source():
    calp=DOC/'S82_PUBLIC_FROZEN_CAL_2026_10_26.json'
    cal=json.loads(calp.read_text());test=load('S82_PUBLIC_FRESH_TEST_2026_10_26.json')
    assert test['frozen_CAL_sha256']==sha256(calp.read_bytes()).hexdigest()
    assert cal['protocol_sha256']==test['protocol_sha256']==s82.digest(s82.PROTOCOL)
    assert cal['operator_sha256']==test['operator_sha256']==s82.digest(s82.OPERATOR)
    assert cal['frozen_S72_CAL_sha256']==s82.s80.LOCKED_S72_SHA256
    assert cal['S80_source_operator_sha256']==s82.digest(s82.s80.OPERATOR)
    assert not cal['smoke_NOT_CONFIRMATORY'] and not test['smoke_NOT_CONFIRMATORY']
    assert all(x['CAL_decision']['CAL_selected']=='baseline' for x in cal['directions'].values())
    assert all(list(x['CAL_only_local_thresholds'].values())==[1.,1.] for x in cal['directions'].values())
    assert test['physical_deployment_authorized'] is False
    assert test['native_GT_count']==0

def test_strict_new_TEST_synthetic_aggregates():
    j=load('S82_PUBLIC_FRESH_TEST_2026_10_26.json')['all_TEST_aggregates']
    expected={'clean':[(40,36,10),(42,43,10),(42,43,10)],
              'blur_sigma_11':[(9,17,2),(11,23,2),(11,23,2)],
              'blur_sigma_16':[(5,19,1),(5,21,1),(5,21,1)]}
    for condition,expect in expected.items():
        assert [(j[condition][arm]['selected_synthetic_correct_links'],
                 j[condition][arm]['synthetic_noncorrect_selected'],
                 j[condition][arm]['complete_three_piece_paths_synthetic']) for arm in s82.ARMS]==expect
        assert all(j[condition][arm]['scenes']==96 for arm in s82.ARMS)

def test_real_images_never_confused_with_physics_truth():
    j=load('S82_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json')
    assert j['tiles']==39 and j['native_physics_GT_count']==0
    assert j['all_machine_joins']=='UNKNOWN' and j['physical_auto_joins_authorized'] is False
    assert [(j['arms'][a]['candidate_lines'],j['arms'][a]['graph_joins_advisory_UNKNOWN'])
             for a in s82.ARMS]==[(83,3),(92,5),(92,5)]

def test_registry_sha_receipts_and_privacy():
    j=load('EXPERIMENT_IDS_MACHINE_ONLY.json')
    records=[x for x in j['experiments'] if x.get('id')==s82.EXPERIMENT_ID]
    assert len(records)==1
    rec=records[0]
    for key in ('protocol','predictor','benchmark','native_audit','calibration','test','native_smoke','report'):
        assert rec[key+'_sha256']==sha256((ROOT/rec[key]).read_bytes()).hexdigest()
    assert rec['native_particle_GT_count']==0 and rec['physical_auto_join_authorized'] is False
    for n in ('S82_PUBLIC_FROZEN_CAL_2026_10_26.json',
              'S82_PUBLIC_FRESH_TEST_2026_10_26.json','S82_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json'):
        raw=(DOC/n).read_text()
        assert all(v not in raw for v in ('points_xy','data:image','_mined00.png','raw_image_base64'))
