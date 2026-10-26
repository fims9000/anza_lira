import json
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/bubble_chamber'

def data(name):return json.loads((DOCS/name).read_text())
def hash_of(path):return sha256(path.read_bytes()).hexdigest()


def test_CAL_already_frozen_before_new_test_and_two_directions():
    cal=data('S84_PUBLIC_FROZEN_CAL_2026_10_26.json')
    fresh=data('S84_PUBLIC_FRESH_TEST_2026_10_26.json')
    assert cal['schema']==fresh['schema'] and cal['phase']=='cal' and fresh['phase']=='test'
    assert fresh['frozen_CAL_sha256']==hash_of(DOCS/'S84_PUBLIC_FROZEN_CAL_2026_10_26.json')
    assert not fresh['smoke_NOT_CONFIRMATORY']
    assert set(fresh['directions'])=={'005_to_006','006_to_005'}
    assert cal['directions']['005_to_006']['CAL_choice']['CAL_only_selected']=='baseline'
    assert cal['directions']['006_to_005']['CAL_choice']['CAL_only_selected']=='s84_photo_q40'
    assert {int(k) for d in fresh['directions'].values() for k in d['TEST_seeds']}=={
        20268403,20268404,20268413,20268414}


def test_fresh_test_metric_accounting_and_no_unsafe_native_claims():
    r=data('S84_PUBLIC_FRESH_TEST_2026_10_26.json')
    for cond,methods in r['all_TEST_aggregates'].items():
        for a,m in methods.items():
            assert m['selected_provisional_links']==m['selected_synthetic_correct_links']+m['synthetic_noncorrect_selected']
            assert m['extra_accepted']+m['extra_rejected']==m['extra_considered']
            assert m['solver_failure']==0 and m['scenes']==96
            assert m['added_pixel_instances']==m['extra_accepted']
            assert m['extra_shape_true_match']+m['extra_shape_decoy_match']<=m['extra_accepted']
        assert methods['baseline']['extra_accepted']==0
    assert r['physical_deployment_authorized'] is False
    assert r['physical_GT_count']==0
    assert r['synthetic_only_metrics'] is True


def test_native_machine_output_is_unknown_and_additive():
    r=data('S84_PUBLIC_NATIVE_UNKNOWN_2026_10_26.json')
    assert r['tiles']==39 and r['all_native_joins']=='UNKNOWN'
    assert r['native_expert_GT_count']==0 and not r['physical_auto_joins_authorized']
    assert r['arms']['baseline']['candidate_lines']==83
    assert r['arms']['s84_photo_q40']['candidate_lines']>=83
    assert r['arms']['s84_context_bridge']['candidate_lines']>=83
    for arm,m in r['arms'].items():
        assert m['candidate_lines']>=83
        assert m['proposed_joins_UNKNOWN']>=0


def test_registry_receipts_are_exact_for_public_s84_files():
    registry=data('EXPERIMENT_IDS_MACHINE_ONLY.json')
    entries=[x for x in registry['experiments'] if x.get('id')=='ANZA-LIRA-BC-S84-20261026']
    assert len(entries)==1
    x=entries[0]
    for file_key,sha_key in [
        ('protocol','protocol_sha256'),('predictor','predictor_sha256'),
        ('benchmark','benchmark_sha256'),('native_audit','native_audit_sha256'),
        ('calibration','calibration_sha256'),('results','results_sha256'),
        ('native_smoke','native_smoke_sha256'),('report','report_sha256')]:
        assert hash_of(ROOT/x[file_key])==x[sha_key]
    assert x['physical_auto_join_authorized'] is False and x['native_GT']==0
    assert x['test_scenes_per_condition']==96
