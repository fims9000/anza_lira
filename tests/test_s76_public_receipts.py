"""S76 public-safe aggregate and experiment registry cannot silently drift."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/bubble_chamber'
RID='ANZA-LIRA-BC-S76-20261026'


def _data():
    return json.loads((DOC/'S76_PUBLIC_FRESH_SHAPE_ENDPOINT_DIAGNOSTIC_2026_10_26.json').read_text())


def test_s76_public_results_exact_provenance_and_no_native_claims():
    data=_data()
    assert data['experiment_id']==RID and data['phase']=='TEST'
    assert not data['smoke_NON_CONFIRMATORY']
    assert data['native_H50Z_GT_count']==0
    assert data['physical_auto_joins_authorized'] is False
    assert data['model_training_or_threshold_selection_performed'] is False
    assert data['S69_official_scoring_unchanged'] is True
    assert data['source_frame_split_sha256']=='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef'
    assert data['seeds']=={'005':[20267603,20267604],'006':[20267613,20267614]}
    assert data['scene_count_per_seed']==24
    assert sum(x['scene_count'] for d in data['directions'].values() for x in (
        next(iter(v['conditions'].values()))['arms']['identity'] for v in d['seed_aggregates'].values()))==96
    for path,key in [('docs/bubble_chamber/S76_FROZEN_ENDPOINT_BLIND_SHAPE_AUDIT_2026_10_26.md','protocol_sha256'),
                     ('bubble_chamber/s76_shape_endpoint_diagnostics.py','auditor_sha256'),
                     ('bubble_chamber/s69_orientation_invariant_audit.py','s69_original_sha256'),
                     ('docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json','S72_calibration_sha256')]:
        assert sha256((ROOT/path).read_bytes()).hexdigest()==data[key]


def test_s76_aggregates_decompose_endpoints_and_joins_exactly():
    data=_data()
    keys=('matched_neighbor_correct_port','matched_neighbor_wrong_port',
          'matched_neighbor_endpoint_unlocalized','matched_wrong_identity_pair',
          'unmatched_or_ambiguous_endpoint_blind')
    for cond,arms in data['aggregate_all_scenes'].items():
        for arm,val in arms.items():
            assert val['scene_count']==96
            assert val['shape_true_fragments_matched']==(
                val['shape_true_both_endpoints_close']+val['shape_true_one_or_both_endpoints_far'])
            assert val['shape_matched_one_or_both_endpoints_far']==(
                val['shape_true_one_or_both_endpoints_far']+val['shape_decoy_one_or_both_endpoints_far'])
            assert val['selected_proposals']==sum(val[k] for k in keys)
            assert val['selected_proposals']==val['official_s69_correct_links']+val['official_s69_noncorrect_links']
            assert val['shape_true_fragments_matched']<=6*96
            assert val['shape_only_true_pair_availability']<=4*96
    # Frozen full-synthetic-only study (never physical track accuracy).
    assert [(data['aggregate_all_scenes'][c]['identity']['shape_true_fragments_matched'],
             data['aggregate_all_scenes'][c]['identity']['shape_true_one_or_both_endpoints_far'],
             data['aggregate_all_scenes'][c]['identity']['official_s69_correct_links'])
            for c in ('clean','blur_sigma_11','blur_sigma_16')]==[(272,38,72),(137,30,18),(116,26,13)]


def test_s76_registry_hashes_and_historical_reports_unmodified():
    data=_data()
    reg=json.loads((DOC/'EXPERIMENT_IDS_MACHINE_ONLY.json').read_text())
    matches=[x for x in reg['experiments'] if x['id']==RID]
    assert len(matches)==1
    entry=matches[0]
    assert entry['native_physics_GT_count']==0 and not entry['physical_auto_join_authorized']
    for key in ('protocol','auditor','benchmark','report','results'):
        assert sha256((ROOT/entry[key]).read_bytes()).hexdigest()==entry[key+'_sha256']
    assert entry['S69_original_sha256']==data['s69_original_sha256']
    assert entry['frozen_S72_calibration_sha256']==data['S72_calibration_sha256']
    assert entry['new_synthetic_scene_seeds']==[20267603,20267604,20267613,20267614]
    assert (DOC/'S75C_POSTHOC_ERROR_TAXONOMY_2026_10_26.md').exists()


def test_s76_result_file_does_not_embed_private_pixel_paths_or_coordinates():
    text=(DOC/'S76_PUBLIC_FRESH_SHAPE_ENDPOINT_DIAGNOSTIC_2026_10_26.json').read_text()
    for unsafe in ('"polylines"','"injected"','"source_track_coordinates"',
                   '"original_photo_pixels"','"raw_image"','/mnt/data/s71_inputs/',
                   '/mnt/data/s76_PRIVATE/','"frame_filename"'):
        assert unsafe not in text
