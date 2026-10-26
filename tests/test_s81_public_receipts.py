import json
from hashlib import sha256
from pathlib import Path
from scripts.research.bubble_chamber import s81_calibrate_quality_gate as s81

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'

def get(name):return json.loads((DOC/name).read_text())


def test_immutable_cal_test_and_baseline_decisions():
    calp=DOC/'S81_PUBLIC_FROZEN_CAL_2026_10_26.json'
    cal=json.loads(calp.read_text())
    test=get('S81_PUBLIC_FRESH_TEST_2026_10_26.json')
    assert test['frozen_CAL_sha256']==sha256(calp.read_bytes()).hexdigest()
    assert cal['protocol_sha256']==test['protocol_sha256']==s81.digest(s81.PROTOCOL)
    assert cal['operator_sha256']==test['operator_sha256']==s81.digest(s81.OPERATOR)
    assert not cal['smoke_NOT_CONFIRMATORY'] and not test['smoke_NOT_CONFIRMATORY']
    assert all(x['CAL_decision']['CAL_selected']=='baseline' for x in cal['directions'].values())
    assert test['native_H50Z_GT']==0 and test['physics_auto_join_authorized'] is False


def test_heldout_synthetic_metrics_are_locked_and_not_physics_claim():
    test=get('S81_PUBLIC_FRESH_TEST_2026_10_26.json')
    d=test['all_TEST_aggregates']
    assert set(d)==set(s81.TEST_CONDITIONS)
    assert all(x['scenes']==96 for arms in d.values() for x in arms.values())
    assert [(d['blur_sigma_11'][a]['selected_synthetic_correct_links'],
             d['blur_sigma_11'][a]['synthetic_noncorrect_selected'],
             d['blur_sigma_11'][a]['quality_gate_triggered'])
           for a in s81.ARMS]==[(19,22,0),(23,22,96),(23,22,96)]
    assert [(d['clean'][a]['selected_synthetic_correct_links'],
             d['clean'][a]['synthetic_noncorrect_selected'])
           for a in s81.ARMS]==[(48,29),(51,30),(51,31)]
    assert [(d['blur_sigma_16'][a]['selected_synthetic_correct_links'],
             d['blur_sigma_16'][a]['synthetic_noncorrect_selected'])
           for a in s81.ARMS]==[(13,17),(14,22),(14,22)]


def test_native_H50Z_predictions_are_unknown():
    r=get('S81_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json')
    assert r['unlabelled_private_tiles']==39 and r['native_particle_GT_count']==0
    assert r['physical_auto_joins_authorized'] is False
    assert r['all_graph_proposals']=='UNKNOWN'
    assert [r['arms'][arm]['quality_gate_triggered'] for arm in s81.ARMS]==[0,25,26]


def test_experiment_registry_immutability_and_privacy():
    r=get('EXPERIMENT_IDS_MACHINE_ONLY.json')
    a=[x for x in r['experiments'] if x.get('id')==s81.EXPERIMENT_ID]
    assert len(a)==1
    item=a[0]
    for k,path in [
        ('algorithm_sha256','bubble_chamber/s81_quality_gated_instances.py'),
        ('benchmark_sha256','scripts/research/bubble_chamber/s81_calibrate_quality_gate.py'),
        ('protocol_sha256','docs/bubble_chamber/S81_FROZEN_PIXEL_QUALITY_GATED_MULTISCALE_2026_10_26.md'),
        ('report_sha256','docs/bubble_chamber/S81_QUALITY_GATE_TRANSFER_NEGATIVE_2026_10_26.md'),
        ('native_audit_sha256','scripts/research/bubble_chamber/s81_native_quality_gate_readonly.py')]:
        assert item[k]==sha256((ROOT/path).read_bytes()).hexdigest()
    assert item['replay_TEST_byte_identical'] is True
    for name in ('S81_PUBLIC_FROZEN_CAL_2026_10_26.json','S81_PUBLIC_FRESH_TEST_2026_10_26.json','S81_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json'):
        raw=(DOC/name).read_text()
        assert all(x not in raw for x in ('points_xy','data:image','_mined00.png','raw_image_base64'))
