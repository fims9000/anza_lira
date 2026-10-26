"""S80 immutable public research receipts, never synthetic GT for inference."""
import json
from pathlib import Path
from hashlib import sha256
from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'

def data(filename):return json.loads((DOC/filename).read_text())


def test_s80_public_cal_before_test_lock_and_no_false_real_claims():
    c=DOC/'S80_PUBLIC_FROZEN_CAL_2026_10_26.json'
    t=DOC/'S80_PUBLIC_FRESH_TEST_2026_10_26.json'
    cal=json.loads(c.read_text());test=json.loads(t.read_text())
    assert test['frozen_S80_CAL_sha256']==sha256(c.read_bytes()).hexdigest()
    assert cal['protocol_sha256']==test['protocol_sha256']==s80.digest(s80.PROTOCOL)
    assert cal['operator_sha256']==test['operator_sha256']==s80.digest(s80.OPERATOR)
    assert not cal['smoke_NOT_CONFIRMATORY'] and not test['smoke_NOT_CONFIRMATORY']
    assert all(d['CAL_decision']['CAL_selected']=='baseline' for d in cal['directions'].values())
    assert test['physical_auto_joins_authorized'] is False
    assert test['native_verified_track_GT']==0
    assert set(test['all_TEST_aggregates'])==set(s80.TEST_CONDITIONS)
    assert all(x['scenes']==96 for arms in test['all_TEST_aggregates'].values() for x in arms.values())


def test_locked_test_correct_synthetic_only_results():
    r=data('S80_PUBLIC_FRESH_TEST_2026_10_26.json')['all_TEST_aggregates']
    assert [(r['blur_sigma_11'][arm]['selected_synthetic_correct_links'],
             r['blur_sigma_11'][arm]['synthetic_noncorrect_selected'],
             r['blur_sigma_11'][arm]['complete_three_piece_paths_synthetic'])
            for arm in s80.ARMS]==[(20,21,5),(24,28,6),(24,20,6)]
    assert [(r['clean'][arm]['selected_synthetic_correct_links'],
             r['clean'][arm]['synthetic_noncorrect_selected'])
             for arm in s80.ARMS]==[(51,41),(53,55),(53,46)]


def test_native_unknown_not_a_physics_accuracy_report():
    j=data('S80_PUBLIC_NATIVE_UNLABELLED_2026_10_26.json')
    assert j['genuine_private_tiles']==39 and j['native_particle_GT']==0
    assert j['native_physical_joins_authorized'] is False
    assert j['all_proposals_status']=='UNKNOWN'
    assert [j['aggregates'][arm]['image_only_line_candidates'] for arm in s80.ARMS]==[83,114,95]


def test_experiment_registry_receipts_and_no_private_data():
    registry=data('EXPERIMENT_IDS_MACHINE_ONLY.json')
    exp=[r for r in registry['experiments'] if r.get('id')==s80.EXPERIMENT_ID]
    assert len(exp)==1
    item=exp[0]
    for field,path in (
        ('algorithm_sha256','bubble_chamber/s80_multiscale_instances.py'),
        ('benchmark_sha256','scripts/research/bubble_chamber/s80_calibrate_then_test_multiscale.py'),
        ('protocol_sha256','docs/bubble_chamber/S80_FROZEN_MULTISCALE_PIXEL_INSTANCES_2026_10_26.md'),
        ('report_sha256','docs/bubble_chamber/S80_MULTISCALE_PIXEL_INSTANCES_CAL_TEST_2026_10_26.md'),
        ('native_audit_code_sha256','scripts/research/bubble_chamber/s80_native_unlabelled_audit.py')):
        assert item[field]==sha256((ROOT/path).read_bytes()).hexdigest()
    assert item['replay_byte_identical'] is True
    for name in ['S80_PUBLIC_FROZEN_CAL_2026_10_26.json','S80_PUBLIC_FRESH_TEST_2026_10_26.json',
                 'S80_PUBLIC_NATIVE_UNLABELLED_2026_10_26.json']:
        raw=(DOC/name).read_text()
        for forbidden in ('points_xy','base64,','_mined00.png','original_bmp_path','native_true_particle_id'):
            assert forbidden not in raw
