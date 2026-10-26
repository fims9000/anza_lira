"""Public-safe S71 retrospective registration, not new physics training."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/bubble_chamber/S71_RETROSPECTIVE_S65_S67_PUBLIC_AGGREGATE_2026_10_26.json'


def test_s71_totals_and_safety_gate():
    x=json.loads(PATH.read_text())
    assert x['new_independent_test_scenes']==0
    assert x['native_verified_H50Z_particle_tracks']==0
    assert not x['physical_auto_joins_authorized']
    assert x['historical_CAL_TEST_exactly_reproduced']
    assert x['S67_historical_paired_CI_exactly_reproduced']
    expected={
     'S65':{'baseline':(15,52,88,51,1,13),'0.7':(4,14,23,13,0,4),'0.4':(2,13,16,5,0,1)},
     'S66':{'baseline':(18,47,73,44,3,11),'ridge_extend_strict':(16,51,91,56,1,12),'ridge_extend_relaxed':(19,55,93,57,1,12)},
     'S67':{'baseline':(8,44,78,42,2,11),'s67_crossing_orientation_v1':(8,43,72,37,2,11)}
    }
    for study,methods in expected.items():
        actual=x['studies'][study]['total_TEST']
        for method,(o,n,wo,wn,po,pn) in methods.items():
            data=actual[method]
            assert (data['historic']['selected_synthetic_correct_links'],
               data['corrected']['selected_synthetic_correct_links'],
               data['historic']['synthetic_noncorrect_selected'],
               data['corrected']['synthetic_noncorrect_selected'],
               data['historic']['complete_three_piece_paths_synthetic'],
               data['corrected']['complete_three_piece_paths_synthetic'])==(o,n,wo,wn,po,pn)


def test_calibration_selection_changed_only_in_s65():
    x=json.loads(PATH.read_text())['studies']
    for direction in ('005_to_006','006_to_005'):
        assert x['S65']['directions'][direction]['CAL_corrected_selected'] is None
        assert x['S65']['directions'][direction]['CAL_choice_changed']
        assert x['S66']['directions'][direction]['CAL_corrected_selected']=='s39_original'
        assert x['S67']['directions'][direction]['CAL_corrected_selected']=='s39_original'
    assert x['S65']['directions']['005_to_006']['CAL_historic_selected']==.7
    assert x['S65']['directions']['006_to_005']['CAL_historic_selected']==.4


def test_no_private_images_tracks_or_artificial_gt_uploaded():
    x=json.loads(PATH.read_text());s=PATH.read_text()
    for forbidden in ('/mnt/data','private_visualmetro','PRIVATE_source_frames','source_nodes',
         'base64,','data:image','\\\\','file_000000'):
        assert forbidden not in s
    assert x['records_are_SYNTHETIC_ONLY'] is True
    # Histories are still present, not overwritten or silently renamed.
    assert (ROOT/'docs/bubble_chamber/S65_PUBLIC_LOCKED_TEST_2026_10_26.json').is_file()
    assert (ROOT/'docs/bubble_chamber/S66_PUBLIC_FRESH_TEST_2026_10_26.json').is_file()
    assert (ROOT/'docs/bubble_chamber/S67_PUBLIC_SYNTHETIC_RESULTS_2026_10_26.json').is_file()
