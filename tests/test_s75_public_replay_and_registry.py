"""Audited S75 CAL-before-TEST, exact replay, and public-only registry provenance."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'


def test_frozen_cal_and_test_aggregate_lock():
    cal=DOC/'S75_PUBLIC_FROZEN_CAL_2026_10_26.json'
    test=DOC/'S75_PUBLIC_FRESH_TEST_2026_10_26.json'
    c=json.loads(cal.read_text());t=json.loads(test.read_text())
    assert c['phase']=='cal' and not c['smoke_NOT_CONFIRMATORY']
    assert t['phase']=='test' and not t['smoke_NOT_CONFIRMATORY']
    assert t['frozen_CAL_SHA256']==sha256(cal.read_bytes()).hexdigest()
    assert t['frozen_S72_calibration_sha256']=='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
    assert set(c['directions'])==set(t['directions'])=={'005_to_006','006_to_005'}
    assert all(x['CAL_only_choice']['chosen_CAL_only']=='identity' for x in c['directions'].values())
    assert all(x['frozen_CAL_preprocess']=='identity' for x in t['directions'].values())
    assert t['physical_H50Z_expert_GT_count']==0
    assert t['real_particle_auto_join_authorized'] is False


def test_test_aggregate_recomputes_from_each_seed():
    t=json.loads((DOC/'S75_PUBLIC_FRESH_TEST_2026_10_26.json').read_text())
    for condition,arms in t['overall_conditions'].items():
        for arm,numbers in arms.items():
            for key,total in numbers.items():
                computed=sum(seed['conditions'][condition]['arms'][arm][key]
                    for direction in t['directions'].values() for seed in direction['TEST_seeds'].values())
                assert computed==total
            assert numbers['scene_count']==96
    assert t['frozen_synthetic_only_stability_gate']['evaluated_direction_condition_cells']==6
    # CAL chose identity. A vacuously passing gate is NOT an improvement claim.
    assert all(d['frozen_CAL_preprocess']=='identity' for d in t['directions'].values())


def test_retrospective_posthoc_exact_file_receipts():
    cal=DOC/'S75_PUBLIC_FROZEN_CAL_2026_10_26.json'
    test=DOC/'S75_PUBLIC_FRESH_TEST_2026_10_26.json'
    for source in ('005','006'):
        p=DOC/f'S75_POSTHOC_PAIRED_CI_{source}_2026_10_26.json'
        a=json.loads(p.read_text())
        assert a['original_S75_TEST_sha256']==sha256(test.read_bytes()).hexdigest()
        assert a['original_S75_CAL_sha256']==sha256(cal.read_bytes()).hexdigest()
        assert a['no_test_set_reselection'] is True and a['independent_new_test'] is False
        assert a['physical_native_ground_truth']==0
        assert len(a['original_TEST_seeds_replayed'])==2
        assert set(a['conditions'])=={'clean','blur_sigma_11','blur_sigma_16'}


def test_registry_hashes_and_no_private_outputs_committed():
    reg=json.loads((DOC/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    entries=[e for e in reg['experiments'] if e['id']=='ANZA-LIRA-BC-S75-20261026']
    assert len(entries)==1
    entry=entries[0]
    for field in ('predictor','benchmark','protocol','report','calibration','results',
                  'posthoc_runner','posthoc_results_005','posthoc_results_006'):
        path=ROOT/entry[field]
        assert path.is_file()
        assert entry[field+'_sha256']==sha256(path.read_bytes()).hexdigest()
    assert entry['native_physics_auto_join_authorized'] is False
    assert entry['native_H50Z_expert_GT_count']==0
    assert entry['CAL_only_selected']=={'005_to_006':'identity','006_to_005':'identity'}
