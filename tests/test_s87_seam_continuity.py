"""No real H50Z pixels, particle identities or hidden operator labels."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp, sha256_file
from bubble_chamber.s87_seam_continuity import assemble, run


def _source(tmp_path, draw):
    gray = np.full((306, 333), 220, np.uint8)
    draw(gray)
    path = tmp_path / 'FAKE_SYNTHETIC.bmp'
    assert cv2.imwrite(str(path), gray)
    return path


def _run_s86(tmp_path, monkeypatch, draw, **kw):
    from bubble_chamber import full_bmp_segmentation as s86
    source = _source(tmp_path, draw)
    # Deterministic, source-blind synthetic-pixel segmentation for seam tests.
    monkeypatch.setattr(s86, 'predict_mask', lambda pixels, cfg, az_model=None: pixels < 80)
    out = tmp_path / 'S86_PRIVATE'
    config = FullBMPConfig(method='contrast', core_size=128, halo=32,
                           preview_limit=0, min_segment_length_px=5, **kw)
    run_full_bmp(source, out, config)
    return source, out


def test_image_contiguous_line_is_traced_through_two_tile_seams(tmp_path, monkeypatch):
    src, inp = _run_s86(tmp_path, monkeypatch,
                        lambda im: cv2.line(im, (12, 70), (317, 70), 30, 1))
    before_sha = sha256_file(inp / 'manifest.json')
    report = run(inp, tmp_path / 'S87_PRIVATE')
    assert report['status'] == 'COMPLETE_MACHINE_ONLY'
    assert report['stitched_image_components'] >= 1
    assert report['unique_tangent_compatible_adjacencies'] >= 2
    assert report['physical_GT_count'] == 0 and report['physical_auto_joins_authorized'] is False
    tracks = [json.loads(s) for s in (tmp_path/'S87_PRIVATE/seam_tracks.ndjson').read_text().splitlines()]
    assert any(len(t['source_fragment_ids_in_geometric_order']) >= 3 for t in tracks)
    for t in tracks:
        points = np.array(t['points_xy_source_px'])
        assert np.max(np.abs(np.diff(points, axis=0))) <= 1
        assert t['physical_identity'] == 'UNKNOWN' and not t['is_particle_track_gt']
        assert all(np.max(np.abs(np.subtract(a, b))) == 1 for a, b in t['seam_observed_pixel_steps_xy'])
    assert sha256_file(inp / 'manifest.json') == before_sha
    assert sha256_file(src) == json.loads((inp / 'manifest.json').read_text())['source_bmp_sha256']
    assert run(inp, tmp_path / 'S87_PRIVATE') == report  # reopen/replay deterministic
    with pytest.raises(FileExistsError):
        run(inp, tmp_path / 'S87_PRIVATE', tangent_angle_deg=40.0)


def test_ambiguous_crossing_abstains_instead_of_merging(tmp_path):
    # Synthetic ordered open fragments at four-way seam; two options for both.
    def frag(fid, points):
        return {'fragment_id': fid, 'centerline_xy_source_px': points, 'status': 'UNKNOWN'}
    rec = [
        {'tile_id': 'A', 'core_xyxy': [0, 0, 128, 128], 'fragments': [
            frag('A1', [[122, 62], [123, 62], [124, 62], [125, 62], [126, 62], [127, 62]]),
            frag('A2', [[122, 63], [123, 63], [124, 63], [125, 63], [126, 63], [127, 63]])]},
        {'tile_id': 'B', 'core_xyxy': [128, 0, 256, 128], 'fragments': [
            frag('B1', [[128, 62], [129, 62], [130, 62], [131, 62], [132, 62], [133, 62]]),
            frag('B2', [[128, 63], [129, 63], [130, 63], [131, 63], [132, 63], [133, 63]])]},
    ]
    tracks, decisions, stats = assemble(rec)
    assert stats['seam_candidate_pairs'] == 4
    assert stats['unique_tangent_compatible_adjacencies'] == 0
    assert stats['stitched_image_components'] == 0
    assert len(decisions) == 4
    assert all(d['decision'] == 'multiple_adjacent_endpoints_ABSTAIN' for d in decisions)


def test_tangent_veto_on_orthogonal_observed_crossing():
    def make(fid, core, pts):
        return {'tile_id': fid, 'core_xyxy': core,
                'fragments': [{'fragment_id': fid+'1', 'centerline_xy_source_px': pts, 'status': 'UNKNOWN'}]}
    left = make('A', [0,0,128,128], [[124,60],[125,60],[126,60],[127,60]])
    right = make('B', [128,0,256,128], [[128,60],[128,61],[128,62],[128,63]])
    tracks, decisions, stats = assemble([left,right])
    assert not tracks and stats['seam_candidate_pairs'] == 1
    assert decisions[0]['decision'] == 'tangent_incompatible_ABSTAIN'


def test_mismatched_or_modified_S86_cannot_be_used(tmp_path, monkeypatch):
    _, inp = _run_s86(tmp_path, monkeypatch,
                       lambda im: cv2.line(im, (12, 70), (317, 70), 30, 1))
    recpath = next((inp/'records').glob('*.json'))
    rec = json.loads(recpath.read_text())
    rec['core_xyxy'][0] += 1
    recpath.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match='tile identity'):
        run(inp, tmp_path/'BAD')


def test_source_mask_tamper_is_rejected(tmp_path, monkeypatch):
    _, inp = _run_s86(tmp_path, monkeypatch, lambda im: cv2.line(im, (12,70), (317,70), 30, 1))
    next((inp/'masks').glob('*.png')).write_bytes(b'bad')
    with pytest.raises(ValueError, match='modified S86 mask shard'):
        run(inp, tmp_path/'BAD')


def test_no_private_output_inside_git_checkout(tmp_path, monkeypatch):
    _, inp = _run_s86(tmp_path, monkeypatch, lambda im: cv2.line(im, (12,70), (317,70), 30, 1))
    import bubble_chamber.s87_seam_continuity as s87
    with pytest.raises(ValueError, match='outside public Git repository'):
        run(inp, Path(s87.__file__).resolve().parents[1]/'BAD_PRIVATE')
    with pytest.raises(ValueError, match='inside frozen S86'):
        run(inp, inp/'nested_private')


def test_private_gallery_uses_verified_original_pixels_and_reopens(tmp_path, monkeypatch):
    src, inp = _run_s86(tmp_path, monkeypatch,
                        lambda im: cv2.line(im, (12, 70), (317, 70), 30, 1))
    out = tmp_path / 'S87_PHYSICIST_VIEW_PRIVATE'
    report = run(inp, out, bmp_path=src, preview_limit=4)
    assert report['private_gallery_crop_count'] >= 1
    assert len(list((out/'preview').glob('*_original.png'))) == report['private_gallery_crop_count']
    assert len(list((out/'preview').glob('*_observed.png'))) == report['private_gallery_crop_count']
    original = cv2.imread(str(next((out/'preview').glob('*_original.png'))), cv2.IMREAD_GRAYSCALE)
    overlay = cv2.imread(str(next((out/'preview').glob('*_observed.png'))), cv2.IMREAD_COLOR)
    assert original.shape == (192, 192)
    assert overlay.shape == (192, 192, 3)
    assert 'PHYSICS UNKNOWN' in (out/'index.html').read_text(encoding='utf-8') or \
           'physical particle' in (out/'index.html').read_text(encoding='utf-8')
    assert (out/'report.json').stat().st_mode & 0o077 == 0
    assert run(inp, out, bmp_path=src, preview_limit=4) == report
    changed = tmp_path/'WRONG.bmp'
    fake = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    fake[10,10] ^= 255
    assert cv2.imwrite(str(changed),fake)
    with pytest.raises(ValueError, match='does not match S86'):
        run(inp, tmp_path/'S87_WRONG', bmp_path=changed)
