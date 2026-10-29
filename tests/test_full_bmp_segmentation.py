"""Only synthetic grayscale BMP; no actual H50Z private input in CI."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from bubble_chamber import full_bmp_segmentation as full


def _input(tmp_path):
    src = tmp_path / 'FAKE.bmp'
    image = np.full((287, 326), 210, np.uint8)
    cv2.line(image, (15, 70), (312, 72), 28, 2)  # crosses two tile seams
    cv2.line(image, (116, 24), (120, 262), 48, 2)
    assert cv2.imwrite(str(src), image)
    return src, image


def test_full_bmp_shards_are_exact_cover_and_resumable(tmp_path):
    source, image = _input(tmp_path)
    sha = full.sha256_file(source)
    cfg = full.FullBMPConfig(method='contrast', core_size=128, halo=32,
                             min_segment_length_px=5, preview_limit=3,
                             full_mask_npy=True)
    out = tmp_path / 'private_out'
    report = full.run_full_bmp(source, out, cfg)
    assert report['status'] == 'COMPLETE_MACHINE_ONLY'
    assert report['tile_count'] == 9
    assert report['native_physics_GT_count'] == 0
    assert report['physical_auto_joins_authorized'] is False
    assert report['candidate_foreground_pixels'] > 0
    assert sha == full.sha256_file(source)
    full_mask = np.load(out / 'full_candidate_mask.npy', mmap_mode='r')
    assert full_mask.shape == image.shape
    mosaic = np.empty_like(image)
    records = []
    for path in sorted((out / 'records').glob('*.json')):
        rec = json.loads(path.read_text())
        x0, y0, x1, y1 = rec['core_xyxy']
        part = cv2.imread(str(out / rec['mask_file']), cv2.IMREAD_GRAYSCALE)
        assert part.shape == (y1-y0, x1-x0)
        mosaic[y0:y1, x0:x1] = part
        records.append(rec)
        for frag in rec['fragments']:
            xy = np.asarray(frag['centerline_xy_source_px'])
            assert np.all((xy[:,0] >= x0) & (xy[:,0] < x1))
            assert np.all((xy[:,1] >= y0) & (xy[:,1] < y1))
            assert frag['status'] == 'UNKNOWN'
    assert np.array_equal(mosaic, full_mask)
    assert len(list((out / 'preview').glob('*_source.png'))) == 3
    assert '<html' in (out / 'index.html').read_text()
    assert (out / 'fragments.ndjson').is_file()
    assert (out / 'seam_advisories.ndjson').is_file()
    second = full.run_full_bmp(source, out, cfg, resume=True)
    assert second['tiles_reused_on_this_run'] == 9
    with pytest.raises(FileExistsError):
        full.run_full_bmp(source, out, cfg)
    with pytest.raises(ValueError, match='identity mismatch'):
        full.run_full_bmp(source, out, full.FullBMPConfig(method='frangi', core_size=128, halo=32), resume=True)


def test_seam_advisories_are_only_observed_pixel_neighbors(tmp_path, monkeypatch):
    source, _ = _input(tmp_path)
    monkeypatch.setattr(full, 'predict_mask', lambda pixels, cfg, az_model=None: pixels < 100)
    out = tmp_path / 'results'
    report = full.run_full_bmp(source, out, full.FullBMPConfig(method='contrast',
                               core_size=128, halo=32, preview_limit=0,
                               min_segment_length_px=5))
    assert report['seam_advisories_UNKNOWN'] > 0
    joins = [json.loads(line) for line in (out / 'seam_advisories.ndjson').read_text().splitlines()]
    assert all(x['status'] == 'UNKNOWN' and not x['physical_auto_join'] for x in joins)
    assert all(x['observed_pixel_adjacency_only'] for x in joins)


def test_checkpoint_required_and_private_output(tmp_path):
    source, _ = _input(tmp_path)
    with pytest.raises(ValueError, match='checkpoint'):
        full.run_full_bmp(source, tmp_path/'az', full.FullBMPConfig(method='azunet',core_size=128,halo=32))
    with pytest.raises(ValueError, match='only valid'):
        full.run_full_bmp(source, tmp_path/'other', full.FullBMPConfig(method='contrast',core_size=128,halo=32),checkpoint=source)
    with pytest.raises(ValueError, match='source repository'):
        full.run_full_bmp(source, Path(full.__file__).resolve().parents[1]/'private_wrong',
                          full.FullBMPConfig(method='contrast',core_size=128,halo=32))
    with pytest.raises(ValueError, match='core_size'):
        full.FullBMPConfig(core_size=97).validate()


def test_azunet_mask_inversion_on_toy_model():
    torch = pytest.importorskip('torch')
    class Toy(torch.nn.Module):
        def forward(self, x):
            return (x - .5) * 100
    gray = np.full((64, 96), 230, np.uint8)
    gray[20:30, 25:60] = 40
    mask = full.predict_mask(gray, full.FullBMPConfig(method='azunet',
                             core_size=128, halo=32), az_model=Toy())
    assert mask.shape == gray.shape and mask.dtype == np.bool_
    assert mask[24, 40] and not mask[4, 4]


def test_skeleton_budget_never_silently_certifies(tmp_path, monkeypatch):
    source, _ = _input(tmp_path)
    monkeypatch.setattr(full, 'skeletonize', lambda mask: np.ones_like(mask, dtype=bool))
    out = tmp_path / 'bounded'
    cfg = full.FullBMPConfig(method='contrast', core_size=128, halo=32, max_skeleton_pixels=100)
    result = full.run_full_bmp(source, out, cfg)
    assert result['tiles_with_fragment_extraction_skipped'] == result['tile_count']
    assert result['candidate_fragments'] == 0
    assert result['candidate_foreground_pixels'] > 0


def test_frangi_full_pipeline_runs_on_synthetic_bmp(tmp_path):
    source, _ = _input(tmp_path)
    out = tmp_path / 'frangi'
    cfg = full.FullBMPConfig(method='frangi', core_size=128, halo=32,
                             preview_limit=1, min_segment_length_px=5)
    report = full.run_full_bmp(source, out, cfg)
    assert report['candidate_foreground_pixels'] > 0
    assert report['native_physics_GT_count'] == 0
    for r in (out / 'records').glob('*.json'):
        tile = json.loads(r.read_text())
        assert tile['mask_sha256'] == full.sha256_file(out / tile['mask_file'])


def test_resume_rejects_changed_mask_shard(tmp_path):
    source, _ = _input(tmp_path)
    out = tmp_path / 'tamper'
    cfg = full.FullBMPConfig(method='contrast', core_size=128, halo=32, preview_limit=0)
    full.run_full_bmp(source, out, cfg)
    image = next((out / 'masks').glob('*.png'))
    image.write_bytes(b'invalid replacement')
    with pytest.raises(ValueError, match='resume mask has changed'):
        full.run_full_bmp(source, out, cfg, resume=True)
