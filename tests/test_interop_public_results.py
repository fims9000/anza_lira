"""Immutable provenance and exact original outcome; no native physics claims."""
from pathlib import Path
from hashlib import sha256
import json
from scripts.research.bubble_chamber import interop_comparison as r
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/bubble_chamber'


def report():return json.loads((DOCS/'S85_INTEROP_PUBLIC_TOY_AGGREGATES_2026_10_28.json').read_text())


def test_report_is_full_not_smoke_and_component_hashes_exact():
    j=report()
    assert j['experiment_id']=='ANZA-LIRA-BC-S85-INTEROP-20261028'
    assert j['mode']=='GENERATED_TOY_ONLY_NOT_H50Z'
    assert j['scenes_per_condition']==j['frozen_design_scenes_per_condition']==20
    assert not j['smoke_NOT_CONFIRMATORY']
    assert j['source_not_film070'] and j['image_backgrounds_not_H50Z']
    assert not j['source_train_calibration_per_detector'] and j['shared_uncalibrated_heuristic']
    assert not j['physical_auto_joins_authorized'] and j['native_GT_count']==0
    assert j['protocol_sha256']==sha256(r.PROTOCOL.read_bytes()).hexdigest()
    for rel,expected in j['component_source_sha256'].items():
        assert sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel


def test_toy_results_are_recorded_without_selecting_a_real_winner():
    j=report()
    known={'clean':(10,4,0,13), 'blur_sigma_11':(5,1,0,13),
           'scratch_grain':(10,2,1,13)}
    for cond,chain_counts in known.items():
        found=[j['results'][cond][det]['global_milp']['complete_three_piece_paths_synthetic']
               for det in j['methods']]
        assert tuple(found)==chain_counts
        for det in j['methods']:
            x=j['results'][cond][det]['global_milp']
            assert x['scenes']==20 and x['solver_failure']==0
            assert 0<=x['complete_three_piece_paths_synthetic']<=40
            assert x['selected_synthetic_correct_links']+x['synthetic_noncorrect_selected']>=0
    lsd=j['results']['clean']['opencv-lsd-v1']
    assert lsd['global_milp']['synthetic_noncorrect_selected']==47
    assert lsd['port_greedy']['synthetic_noncorrect_selected']==21
    assert j['paired_global_vs_greedy_chain']['clean']['opencv-lsd-v1']['scene_bootstrap_95pct']==[0.,.15]


def test_research_registry_does_not_assert_real_physics_success():
    registry=json.loads((DOCS/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    entries=[x for x in registry['experiments'] if x.get('id')==r.EXPERIMENT_ID]
    assert len(entries)==1
    rec=entries[0]
    assert rec['native_GT_count']==0 and rec['physical_auto_join_authorized'] is False
    assert not rec['real_H50Z_test_executed'] and rec['detector_specific_calibration_proven'] is False
    for k in ('protocol','predictor','detectors','calibration','benchmark','result','readme','results_report'):
        assert sha256((ROOT/rec[k]).read_bytes()).hexdigest()==rec[k+'_sha256']
