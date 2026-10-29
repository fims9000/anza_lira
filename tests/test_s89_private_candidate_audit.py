"""Synthetic-input only. No actual H50Z pixels or human track labels in Git."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp, sha256_file
from bubble_chamber.s88_generated_benchmark import scene
from bubble_chamber.s88_weak_ridges import WeakRidgeConfig
from bubble_chamber.s89_private_candidate_audit import (
    AuditConfig, SCHEMA, audit_one_fragment, persistence_under_perturbations,
    run, select_tiles, transverse_darkness,
)


def _source(tmp_path):
    item = scene(9090, 'blurred_scratches')
    path = tmp_path / 'SYNTHETIC_NO_H50Z.bmp'
    assert cv2.imwrite(str(path), item.gray)
    out = tmp_path / 'PRIVATE_S86'
    cfg = FullBMPConfig(method='s88-steerable', core_size=128, halo=32,
                        s88_low_snr=.816, s88_high_snr=1.7,
                        min_segment_length_px=5, preview_limit=0)
    data = run_full_bmp(path, out, cfg)
    assert data['candidate_fragments'] > 1
    return path, out


def test_s89_private_end_to_end_with_source_integrity_and_reopen(tmp_path):
    bmp, s86 = _source(tmp_path)
    source_manifest_hash = sha256_file(s86/'manifest.json')
    mask_hashes = {f.name: sha256_file(f) for f in (s86/'masks').glob('*.png')}
    out = tmp_path / 'PRIVATE_S89'
    cfg = AuditConfig(max_tiles=4, preview_tiles=2)
    result = run(s86, bmp, out, cfg)
    assert result['schema'] == SCHEMA
    assert result['sampled_tile_count'] == 4
    assert result['source_total_tiles'] == 4
    assert result['sampled_fragment_count'] > 0
    assert result['physical_auto_accept_count'] == 0
    assert result['physical_auto_reject_count'] == 0
    assert result['physics_track_GT_count'] == 0
    assert result['physical_auto_joins_authorized'] is False
    assert result['no_physics_labels_or_reference_track_masks_used']
    assert (out/'index.html').is_file()
    assert len(list((out/'preview').glob('*_gray.png'))) == 2
    source_records=[json.loads(p.read_text()) for p in (s86/'records').glob('*.json')]
    low_and_high={r['tile_id'] for r in select_tiles(source_records,2)}
    actual_preview={p.name[:-len('_gray.png')] for p in (out/'preview').glob('*_gray.png')}
    assert actual_preview==low_and_high  # preview must show sparse AND dense tiles
    rows = [json.loads(s) for s in (out/'candidate_audit.ndjson').read_text().splitlines()]
    assert len(rows) == result['sampled_fragment_count']
    assert all(r['status'] == 'UNKNOWN' and not r['is_particle_or_artifact_classified'] for r in rows)
    assert all(not r['physical_auto_accept'] and not r['physical_auto_reject'] for r in rows)
    assert all(0 <= r['minimum_image_perturbation_support'] <= 1 for r in rows)
    assert all(r['photometry']['interpretation'] == 'LOCAL_PIXEL_BRIGHTNESS_NOT_DEDX_OR_PARTICLE_ID' for r in rows)
    assert 'NOT verified' in (out/'index.html').read_text(encoding='utf-8')
    assert (out/'report.json').stat().st_mode & 0o077 == 0
    assert (out/'preview').stat().st_mode & 0o077 == 0
    assert result == run(s86, bmp, out, cfg)
    assert source_manifest_hash == sha256_file(s86/'manifest.json')
    assert mask_hashes == {f.name: sha256_file(f) for f in (s86/'masks').glob('*.png')}
    with pytest.raises(FileExistsError, match='existing output differs'):
        run(s86, bmp, out, AuditConfig(max_tiles=4, preview_tiles=2,
                                       unstable_support_review=.9))


def test_s89_wrong_source_bmp_or_source_mask_is_rejected(tmp_path):
    bmp, s86 = _source(tmp_path)
    altered = tmp_path/'WRONG.bmp'
    gray = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
    gray[9,9] ^= 200
    assert cv2.imwrite(str(altered), gray)
    with pytest.raises(ValueError, match='sha256'):
        run(s86, altered, tmp_path/'BAD')
    assert not (tmp_path/'BAD').exists()
    mask = next((s86/'masks').glob('*.png'))
    mask.write_bytes(b'broken')
    with pytest.raises(ValueError, match='modified S86 mask'):
        run(s86, bmp, tmp_path/'BAD2')
    assert not (tmp_path/'BAD2').exists()


def test_s89_refuses_non_s88_source_and_public_destinations(tmp_path):
    bmp, s86 = _source(tmp_path)
    import bubble_chamber.s89_private_candidate_audit as module
    with pytest.raises(ValueError, match='outside public source'):
        run(s86, bmp, Path(module.__file__).resolve().parents[1]/'BAD')
    with pytest.raises(ValueError, match='cannot overwrite S86'):
        run(s86, bmp, s86/'nested')
    # S89 must not silently use non-S88 thresholds to certify consistency.
    manifest = s86/'manifest.json'
    data = json.loads(manifest.read_text())
    data['config']['method'] = 'contrast'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='requires source S86 method'):
        run(s86, bmp, tmp_path/'BAD3')


def test_s89_subpixel_and_image_darkness_are_not_particle_labels():
    dark = np.full((112,128),210,np.uint8)
    cv2.line(dark, (10,56),(116,56),190,2)
    path = np.array([[float(i),56.0] for i in range(16,111)], dtype=float)
    measurement = transverse_darkness(dark,path,flank=4)
    assert measurement['median_transverse_darkness_gray_levels'] > 5
    assert measurement['fraction_darker_than_flanks']>.8
    constant = np.full_like(dark,200)
    baseline=transverse_darkness(constant,path,flank=4)
    assert baseline['median_transverse_darkness_gray_levels']==0
    assert baseline['fraction_darker_than_flanks']==0


def test_s89_constant_stability_and_uncertainty_status():
    gray = np.full((112,128),210,np.uint8)
    score = WeakRidgeConfig(low_snr=.816,high_snr=1.7)
    cfg = AuditConfig()
    blur, grain = persistence_under_perturbations(gray,score,cfg,9001)
    assert not blur.any()
    assert not grain.any()  # Random grain remains below the frozen threshold in this control.
    frag = {'fragment_id':'y000000_x000000_f0000',
            'centerline_xy_source_px':[[int(x),56] for x in range(14,104)],
            'observed_length_px':89.0,
            'touches_tile_boundary':False}
    item = audit_one_fragment(frag,gray,(0,0,128,112),(blur,grain),cfg)
    assert item['minimum_image_perturbation_support']==0
    assert 'PERTURBATION_UNSTABLE_REVIEW' in item['review_flags']
    assert item['physical_auto_reject'] is False
    assert item['is_particle_or_artifact_classified'] is False


def test_s89_stratified_sampling_is_deterministic_and_contains_low_and_high():
    rows = [dict(tile_id=f't{i}',core_xyxy=[i,0,i+1,1],
                 foreground_pixel_count=i,fragment_count=i) for i in range(9)]
    sample = select_tiles(rows,3)
    assert [r['tile_id'] for r in sample]==['t0','t4','t8']
    assert select_tiles(rows,3)==sample
    with pytest.raises(ValueError,match='count'):
        select_tiles(rows,0)


def test_s89_bad_configs_and_mismatched_fragment_coordinates():
    with pytest.raises(ValueError,match='max_tiles'):
        AuditConfig(max_tiles=0).validate()
    with pytest.raises(ValueError,match='flank_distance'):
        AuditConfig(flank_distance_px=30).validate()
    with pytest.raises(ValueError,match='stability_radius'):
        AuditConfig(stability_radius_px=0).validate()
    frag={'fragment_id':'y000000_x000000_f0000',
          'centerline_xy_source_px':[[i,20] for i in range(120,125)],
          'observed_length_px':4.0,'touches_tile_boundary':False}
    gray=np.full((64,64),210,np.uint8)
    mask=(np.zeros_like(gray,bool),np.zeros_like(gray,bool))
    with pytest.raises(ValueError,match='outside recorded read window'):
        audit_one_fragment(frag,gray,(0,0,64,64),mask,AuditConfig())


def test_s89_hard_negative_generator_does_not_mislabeled_scratches():
    from bubble_chamber.s89_generated_stress import VARIANTS,generate,SCHEMA,benchmark
    assert 'scratches_only' in VARIANTS and 'fiducials_only' in VARIANTS
    gray,truth,has_lines=generate(9000,'scratches_only')
    assert gray.shape == (224,256)
    assert not truth.any() and not has_lines
    gray2,truth2,with_lines=generate(9000,'faint_curves')
    assert with_lines and truth2.any()
    assert not np.array_equal(gray,gray2)
    with pytest.raises(ValueError,match='independent'):
        benchmark(1)


def test_s89_small_generated_negative_controls_keep_physical_claims_disabled():
    from bubble_chamber.s89_generated_stress import SCHEMA,VARIANTS,benchmark
    result=benchmark(2)
    assert result['schema']==SCHEMA
    assert result['protocol']['s89_reuses_thresholds_without_TEST_tuning']
    assert result['native_physics_ground_truth_count']==0
    assert result['physical_track_recall_or_precision_measured'] is False
    assert result['physical_auto_accept_or_reject_authorized'] is False
    assert result['s89_changes_original_S86_detector_masks'] is False
    assert set(result['per_variant'])==set(VARIANTS)
    for variant in ('scratches_only','fiducials_only'):
        assert result['per_variant'][variant]['generated_has_injected_lines'] is False
        assert result['per_variant'][variant]['original_s88_candidates'][
            'synthetic_injected_line_coverage_tolerance_2px'] is None
    assert result['per_variant']['faint_curves']['generated_has_injected_lines']
