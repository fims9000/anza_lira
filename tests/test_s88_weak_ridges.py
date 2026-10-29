"""S88 generated-only engineering tests. Never import native film pixels/marks."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bubble_chamber.s88_weak_ridges import WeakRidgeConfig, dark_ridge_candidates
from bubble_chamber.full_bmp_segmentation import FullBMPConfig, run_full_bmp
from bubble_chamber.s88_generated_benchmark import (_predict, _prepare_response, _threshold_response,
                                                      VARIANTS, GRIDS, SCHEMA,
                                                      TARGET_BUDGET, benchmark,
                                                      metrics, scene)


def test_s88_curved_dark_line_with_subpixel_and_no_calibrated_claims():
    img = np.full((200, 240), 203, dtype=np.uint8)
    pts = np.array([[x, round(65+18*np.sin(x/30))] for x in range(12,228)],dtype=np.int32)
    cv2.polylines(img, [pts], False, 187, 2, cv2.LINE_AA)
    mask, d = dark_ridge_candidates(img)
    assert mask.shape == img.shape and mask.dtype == np.bool_
    assert np.count_nonzero(mask) > 110
    center_support = np.mean([mask[max(0,y-2):y+3,x-2:x+3].any() for x,y in pts[::4]])
    assert center_support >= 0.75
    assert d['semantic_status'] == 'MACHINE_CANDIDATES_ONLY_NOT_PHYSICAL_GT'
    assert np.isfinite(d['score_snr_proxy']).all()
    assert np.max(d['score_snr_proxy']) > 0
    assert d['subpixel_xy_px'].shape == (*img.shape,2)
    assert d['tangent_xy'].shape == (*img.shape,2)
    assert np.all(d['selected_sigma_px'][mask] > 0)


def test_s88_constant_bright_and_large_unobserved_gap():
    blank = np.full((144,144),210,dtype=np.uint8)
    mask,diag=dark_ridge_candidates(blank)
    assert not mask.any() and np.max(diag['score_snr_proxy']) == 0
    white=blank.copy()
    cv2.line(white,(15,70),(125,70),250,2)
    bright,_=dark_ridge_candidates(white)
    assert not np.any(bright[68:73,20:120])  # dark-ridge detector, not bright-line detector
    divided=blank.copy()
    cv2.line(divided,(10,55),(60,55),165,1)
    cv2.line(divided,(85,55),(133,55),165,1)
    gap,_=dark_ridge_candidates(divided)
    assert not np.any(gap[53:58,70:76])  # no invented 25-pixel bridge


def test_s88_config_and_shape_rejections():
    with pytest.raises(ValueError,match='sigmas'):
        WeakRidgeConfig(sigmas=(2.4,1.0)).validate()
    with pytest.raises(ValueError,match='low_snr'):
        WeakRidgeConfig(low_snr=3,high_snr=2).validate()
    with pytest.raises(ValueError,match='max_tangent'):
        WeakRidgeConfig(max_tangent_curvature_ratio=1).validate()
    with pytest.raises(ValueError,match='uint8'):
        dark_ridge_candidates(np.zeros((32,32),dtype=np.float32))
    with pytest.raises(ValueError,match='uint8'):
        dark_ridge_candidates(np.zeros((4,8),dtype=np.uint8))
    with pytest.raises(ValueError,match='high_snr'):
        FullBMPConfig(method='s88-steerable',s88_high_snr=.5).validate()


def test_s88_full_bmp_private_pipeline_and_resume(tmp_path):
    s=scene(13,'faint_grain')
    src=tmp_path/'ARTIFICIAL.bmp'
    assert cv2.imwrite(str(src),s.gray)
    cfg=FullBMPConfig(method='s88-steerable',core_size=128,halo=32,
                      s88_low_snr=.9,s88_high_snr=1.8,
                      min_segment_length_px=5,preview_limit=1,full_mask_npy=True)
    out=tmp_path/'private_result'
    result=run_full_bmp(src,out,cfg)
    assert result['status']=='COMPLETE_MACHINE_ONLY'
    assert result['native_physics_GT_count']==0
    assert result['physical_auto_joins_authorized'] is False
    assert result['candidate_foreground_pixels'] > 0
    full=np.load(out/'full_candidate_mask.npy',mmap_mode='r')
    assert full.shape==s.gray.shape
    assert np.isin(np.unique(full),[0,255]).all()
    assert result['tile_count']==4
    assert (out/'index.html').is_file()
    assert (out/'fragments.ndjson').is_file()
    fragments=[json.loads(line) for line in (out/'fragments.ndjson').read_text().splitlines()]
    assert fragments
    for fragment in fragments:
        if fragment['centerline_xy_source_px']:
            sub=np.asarray(fragment['subpixel_centerline_xy_source_px'])
            integer=np.asarray(fragment['centerline_xy_source_px'])
            assert sub.shape==integer.shape
            assert np.max(np.linalg.norm(sub-integer,axis=1)) <= 1.0
            assert fragment['subpixel_semantics']=='LOCAL_IMAGE_GEOMETRY_ONLY_NOT_CALIBRATED_PARTICLE_TRACK'
            assert fragment['s88_ridge_score_proxy_median']>0
    assert run_full_bmp(src,out,cfg,resume=True)['tiles_reused_on_this_run']==4
    with pytest.raises(ValueError,match='identity mismatch'):
        run_full_bmp(src,out,FullBMPConfig(method='frangi',core_size=128,halo=32),resume=True)


def test_s88_generated_protocol_is_disjoint_and_uses_cal_only():
    assert 's88-steerable' in GRIDS
    assert TARGET_BUDGET > 0
    a=scene(1001,VARIANTS[0]);b=scene(1001,VARIANTS[0]);c=scene(5001,VARIANTS[0])
    assert np.array_equal(a.gray,b.gray)
    assert not np.array_equal(a.gray,c.gray)
    m=metrics(np.zeros_like(a.injected_centerline),a.injected_centerline)
    assert m['injected_centerline_tolerant_recall']==0
    assert m['candidate_pixel_fraction']==0
    with pytest.raises(ValueError,match='independent'):
        benchmark(cal_count=1,test_count=2)


def test_s88_small_generated_benchmark_is_source_only_and_reproducible():
    out=benchmark(cal_count=2,test_count=2)
    assert out['schema']==SCHEMA
    assert out['config']['cal_seed_start']!=out['config']['test_seed_start']
    assert out['native_physics_ground_truth_count']==0
    assert out['native_physical_track_recall_measured'] is False
    assert out['physical_auto_joins_authorized'] is False
    assert set(out['paired_generated_TEST_differences']) == {'s88_vs_contrast','s88_vs_frangi'}
    assert len(out['generated_TEST_scene_rows']['s88-steerable']) == 4
    assert set(out['selected_using_CAL_only']) == {'contrast','frangi','s88-steerable'}
    for method,chosen in out['selected_using_CAL_only'].items():
        assert chosen['high_or_threshold'] in GRIDS[method]
        assert method in out['independent_generated_TEST']
        for score in out['independent_generated_TEST'][method]['summary'].values():
            assert 0<=score<=1
        assert len(out['independent_generated_TEST'][method]['95pct_bootstrap_centerline_recall_by_scene'])==2


def test_cal_cached_masks_identical_to_deployed_inference():
    from bubble_chamber.s88_generated_benchmark import scene
    generated=scene(1000,'faint_grain')
    for detector, thresholds in GRIDS.items():
        response=_prepare_response(generated.gray,detector)
        for threshold in thresholds:
            assert np.array_equal(_predict(generated.gray,detector,threshold),
                                  _threshold_response(response,detector,threshold))


def test_s88_halo_preserves_candidate_mask_at_inner_tile_seam():
    from bubble_chamber.full_bmp_segmentation import predict_mask
    s=scene(1002,'blurred_scratches')
    cfg=FullBMPConfig(method='s88-steerable',core_size=128,halo=32,
                      s88_low_snr=.816,s88_high_snr=1.7)
    entire=predict_mask(s.gray,cfg)
    half=np.zeros_like(entire)
    for y in (0,128):
        for x in (0,128):
            x1=min(x+128,s.gray.shape[1]);y1=min(y+128,s.gray.shape[0])
            wx=max(0,x-cfg.halo);wy=max(0,y-cfg.halo)
            wx1=min(x1+cfg.halo,s.gray.shape[1]);wy1=min(y1+cfg.halo,s.gray.shape[0])
            tile=predict_mask(s.gray[wy:wy1,wx:wx1],cfg)
            half[y:y1,x:x1]=tile[y-wy:y1-wy,x-wx:x1-wx]
    seam=np.zeros_like(entire)
    seam[:,126:130]=True
    seam[126:130,:]=True
    assert np.count_nonzero(entire[seam]) > 0
    assert np.mean(half[seam]!=entire[seam]) < .04


def test_s88_real_detector_to_s87_observed_pixel_continuity(tmp_path):
    """Integration test: no monkeypatched detector or oracle link identities."""
    from bubble_chamber.s87_seam_continuity import run as run_s87
    gray=np.full((306,333),215,np.uint8)
    cv2.line(gray,(12,71),(319,71),198,1,cv2.LINE_AA)
    bmp=tmp_path/'SYNTHETIC_NOT_REAL.bmp'
    assert cv2.imwrite(str(bmp),gray)
    cfg=FullBMPConfig(method='s88-steerable',core_size=128,halo=32,
                      s88_low_snr=.816,s88_high_snr=1.7,
                      min_segment_length_px=5,preview_limit=0)
    r86=run_full_bmp(bmp,tmp_path/'source_s86',cfg)
    assert r86['candidate_fragments']>=3
    r87=run_s87(tmp_path/'source_s86',tmp_path/'source_s87',
                bmp_path=bmp,preview_limit=0)
    assert r87['unique_tangent_compatible_adjacencies']>=2
    assert r87['stitched_image_components']>=1
    assert r87['physical_auto_joins_authorized'] is False
    tracks=[json.loads(s) for s in (tmp_path/'source_s87/seam_tracks.ndjson').read_text().splitlines()]
    assert any(len(t['source_fragment_ids_in_geometric_order'])>=3 for t in tracks)
    assert all(t['physical_identity']=='UNKNOWN' and not t['is_particle_track_gt'] for t in tracks)
