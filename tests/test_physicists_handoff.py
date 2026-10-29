"""Regression and provenance checks for the PUBLIC, synthetic-safe physics brief."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import pytest

from scripts.research.bubble_chamber import build_physicists_handoff as handoff


ROOT=Path(__file__).resolve().parents[1]


def test_provenance_matches_exact_original_public_json_receipts():
    summary=handoff.load_results()
    assert summary['source_branch']=='research/bubble-chamber-graph-lira'
    assert len(summary['source_file_sha256'])==8
    for name,digest in summary['source_file_sha256'].items():
        assert sha256((ROOT/'docs/bubble_chamber'/name).read_bytes()).hexdigest()==digest
    assert summary['physical_runs']==1
    assert summary['original_photographs']==2
    assert summary['H50Z_selected_512px_crops']==39
    assert summary['visualMetro_train_cal_test_whole_frames']=={'train':79,'cal':30,'test':36}
    assert summary['visualMetro_is_NOT_H50Z_photo_GT'] is True


def test_real_photo_counts_are_not_mislabeled_as_physics_track_counts():
    summary=handoff.load_results()
    got=summary['real_H50Z_machine_only']
    assert (got['old_line_proposals'],got['additional_ridge_proposals'],got['total_line_proposals_not_tracks'])==(133,434,567)
    assert (got['sampled_nonselect_full_frame_tiles'],got['full_frame_tiles_with_machine_proposals'],got['full_frame_line_proposals'])==(128,127,1166)
    assert (got['after_NMS_graph_nodes'],got['possible_join_proposals'],got['selected_join_hypotheses_UNKNOWN'])==(511,182,37)
    assert got['machine_connected_components']==35 and got['components_of_two_lines']==34 and got['components_of_four_lines']==1
    assert summary['expert_verified_H50Z_line_masks']==0
    assert summary['expert_verified_H50Z_same_track_pairs']==0
    assert summary['expert_verified_H50Z_particle_ids']==0
    assert summary['native_GT_count']==0
    assert summary['physical_auto_joins_authorized'] is False
    assert summary['real_native_accuracy_precision_recall_available'] is False


def test_synthetic_only_metrics_and_negative_controls_are_not_promoted():
    got=handoff.load_results()['synthetic_injection_only']
    s28=got['S28_known_anchor_curve_coverage']
    assert s28['curvature_constrained_evidence']=={'fraction':0.96666667,'successes':16,'samples':16}
    assert s28['original_AZ_plus_curvature']['successes']==15
    assert got['S56_oracle_geometry_not_pixel_end_to_end']['scene_count']==256
    assert got['S56_oracle_geometry_not_pixel_end_to_end']['arms']['global_forest']=={'correct':246,'other':13,'full_chains':66}
    assert got['S56_oracle_geometry_not_pixel_end_to_end']['arms']['port_greedy']=={'correct':441,'other':36,'full_chains':129}
    p=got['S72_pixel_to_graph_synthetic_TEST']
    assert p['S39_original']=={'correct':56,'other':52,'full_chains':12}
    assert p['frozen_CAL_selected']=={'correct':56,'other':41,'full_chains':13}
    assert p['possible_three_piece_chains']==192
    assert got['S84_additional_candidate_CAL_selected_failed_safety']['clean']['selected_other']==42
    assert got['S85_detector_swaps_generated_only']['safe_improvement_proven'] is False
    assert set(got['S85_detector_swaps_generated_only']['CAL_selected_per_detector'].values())=={'none'}


def test_build_is_deterministic_byte_exact_and_every_new_file_exists():
    generated=handoff.files_to_generate()
    assert len(generated)==9
    for relative,content in generated.items():
        p=ROOT/relative
        assert p.is_file() and p.read_bytes()==content.encode('utf8')
        assert not any(p.name.lower().endswith(ext) for ext in ('.bmp','.png','.jpg','.pt','.pth','.7z','.zip','.pkl','.npy'))
    assert json.loads(generated['docs/bubble_chamber/physicists/results_manifest.json'])==handoff.load_results()


def test_all_schematic_figures_are_svg_and_marked_nonphysical():
    for name in ('raw','candidates','graph','mask'):
        raw=(ROOT/f'docs/bubble_chamber/physicists/figures/{name}.svg').read_text()
        element=ET.fromstring(raw)
        assert element.tag=='{http://www.w3.org/2000/svg}svg'
        assert 'НЕ ФОТО И НЕ ВЫВОД МОДЕЛИ H50Z' in raw
        assert ('ИСКУССТВЕНН' in raw or 'Искусственн' in raw)
        assert '<image ' not in raw and 'http://' not in raw.replace('http://www.w3.org/2000/svg','')
        assert 'xlink:href' not in raw
        assert len(raw.encode('utf8'))<25000
    graph=(ROOT/'docs/bubble_chamber/physicists/figures/graph.svg').read_text()
    assert 'UNKNOWN' in graph and 'stroke-dasharray' in graph


def test_report_links_resolve_inside_repository_and_no_private_photos():
    rels=['PHYSICISTS_HANDOFF.md','docs/bubble_chamber/PHYSICISTS_RESULTS_2026_10_29.md','docs/bubble_chamber/physicists/README.md']
    for relative in rels:
        p=ROOT/relative
        text=p.read_text()
        for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',text):
            if '://' in link:continue
            resolved=(p.parent/link).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), (p,link)
            assert resolved.is_file(),(p,link)
    brief=(ROOT/'docs/bubble_chamber/PHYSICISTS_RESULTS_2026_10_29.md').read_text()
    assert 'RESEARCH / READ-ONLY' in brief
    assert '0** экспертных масок' in brief
    assert 'только схемы, не реальные H50Z' in brief.lower() or 'схемы, не реальные H50Z' in brief
    assert '0 соединений на TEST' in brief
    assert '567 физических треков' in brief  # Prohibited as an explicit negative quote, not a claim.
    for forbidden in ('data:image/jpeg;base64,','<base64>','raw_source_points_xy','x0_y0_native_real_coordinates'):
        assert forbidden not in brief


def test_offline_html_has_no_downloads_external_dependencies_or_native_claims():
    content=(ROOT/'docs/bubble_chamber/physicists/demo.html').read_text()
    assert '<meta charset="utf-8">' in content
    assert 'НЕ ФОТО И НЕ ВЫВОД МОДЕЛИ H50Z' not in content or 'не являются фотографиями' in content
    assert 'не являются фотографиями' in content
    assert 'UNKNOWN' in content
    assert '192' in content and '52' in content and '41' in content
    assert 'fetch(' not in content and 'XMLHttpRequest' not in content and '<iframe' not in content
    assert '<img ' in content and 'figures/graph.svg' in content
    assert 'https://' not in content and 'http://' not in content
    for name in ('raw','candidates','graph','mask'):
        assert f'data-panel="{name}"' in content
        assert f'id="panel-{name}"' in content


def test_source_drift_causes_fail_closed_failure_before_creating_outputs(tmp_path):
    # No silent correction to user research if the original S39 counters changed.
    root=tmp_path/'fake_source';dir=root/'docs/bubble_chamber';dir.mkdir(parents=True)
    for relative in handoff.SRC.values():
        shutil.copy2(ROOT/'docs/bubble_chamber'/relative,dir/relative)
    original=json.loads((dir/'S39_PUBLIC_RESULTS.json').read_text())
    original['real_H50Z_selected_machine_candidates']['all_machine_proposals']=566
    (dir/'S39_PUBLIC_RESULTS.json').write_text(json.dumps(original))
    with pytest.raises(ValueError,match='567 machine line'):
        handoff.load_results(root)
    assert not (dir/'physicists').exists()
