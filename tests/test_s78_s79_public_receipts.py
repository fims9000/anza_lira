"""Frozen public S78/S79 decisions and SHA receipts, no private imagery."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/bubble_chamber'


def get(name):return json.loads((DOCS/name).read_text(encoding='utf8'))
def hash_path(path):return sha256(Path(path).read_bytes()).hexdigest()


def test_public_cal_decision_locked_to_saved_aggregates():
    for phase in ('78','79'):
        cal_path=DOCS/f'S{phase}_PUBLIC_FROZEN_CAL_2026_10_26.json'
        test=get(f'S{phase}_PUBLIC_FRESH_TEST_2026_10_26.json')
        cal=get(cal_path.name)
        assert cal['phase']=='cal' and test['phase']=='test'
        assert cal['protocol_sha256']==test['protocol_sha256']
        assert test[f'S{phase}_frozen_CAL_SHA256']==hash_path(cal_path)
        assert cal['source_frame_split_sha256']==test['source_frame_split_sha256']
        assert test['native_H50Z_GT_count']==0 and not test['native_physical_auto_joins_authorized']
        assert set(test['directions'])=={'005_to_006','006_to_005'}
        assert all(v['CAL_selected']=='baseline' for v in test['directions'].values())
        assert all(len(v['test_seeds'])==2 for v in test['directions'].values())


def test_test_s78_no_actual_extra_alternatives_or_changes_to_graph():
    j=get('S78_PUBLIC_FRESH_TEST_2026_10_26.json')
    for cond,arms in j['all_TEST_aggregates'].items():
        original=arms['baseline']
        for method in ('alternative_strict','alternative_relaxed'):
            x=arms[method]
            assert x['candidate_extra_alternatives']==0
            assert x['selected_pixel_alternatives']==0
            for k in ('selected_synthetic_correct_links','synthetic_noncorrect_selected','complete_three_piece_paths_synthetic'):
                assert x[k]==original[k]
    assert j['all_TEST_aggregates']['clean']['baseline']['selected_synthetic_correct_links']==62
    assert j['all_TEST_aggregates']['clean']['baseline']['complete_three_piece_paths_synthetic']==16


def test_test_s79_rescue_introduced_no_true_joins_and_more_clean_wrong():
    j=get('S79_PUBLIC_FRESH_TEST_2026_10_26.json')
    clean=j['all_TEST_aggregates']['clean'];base=clean['baseline']
    assert (base['selected_synthetic_correct_links'],base['synthetic_noncorrect_selected'],base['complete_three_piece_paths_synthetic'])==(48,45,13)
    assert clean['rescue_strict']['selected_synthetic_correct_links']==48
    assert clean['rescue_strict']['synthetic_noncorrect_selected']==46
    assert clean['rescue_relaxed']['synthetic_noncorrect_selected']==47
    for cond in ('blur_sigma_11','blur_sigma_16'):
        rows=j['all_TEST_aggregates'][cond]
        for method in ('rescue_strict','rescue_relaxed'):
            assert rows[method]['selected_synthetic_correct_links']==rows['baseline']['selected_synthetic_correct_links']
            assert rows[method]['synthetic_noncorrect_selected']==rows['baseline']['synthetic_noncorrect_selected']


def test_native_smoke_is_machine_only_no_independent_run_claim():
    j=get('S78_S79_PUBLIC_NATIVE_UNLABELLED_SMOKE_2026_10_26.json')
    assert j['original_H50Z_crop_count']==39 and j['actual_independent_photographic_runs']==1
    assert j['native_H50Z_GT_count']==0 and not j['physical_join_authorization']
    for data in j['aggregated_machine_counts'].values():
        assert data['machine_curves']==83
        assert data['advisory_joins_UNKNOWN']==3


def test_registry_provenance_and_no_private_paths():
    x=get('EXPERIMENT_IDS_MACHINE_ONLY.json')
    for phase in ('78','79'):
        entries=[r for r in x['experiments'] if r['id']==f'ANZA-LIRA-BC-S{phase}-20261026']
        assert len(entries)==1
        entry=entries[0]
        assert entry['native_H50Z_GT_count']==0 and not entry['physical_auto_joins_authorized']
        assert entry['synthetic_TEST_scenes']==96 and entry['TEST_replay_byte_identical']
        for k in ('protocol','predictor','benchmark','calibration','test','report'):
            assert hash_path(ROOT/entry[k])==entry[k+'_sha256']
            assert '/PRIVATE/' not in entry[k]
