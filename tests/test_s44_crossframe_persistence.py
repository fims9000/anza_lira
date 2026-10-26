import cv2
import numpy as np
import pytest
from bubble_chamber.s44_crossframe_persistence import (
    fit_common_fiducials, ridge_near_path, compare_registered_ridges,
)


def fake_labels(offset=(-17.,29.)):
    positions = [(1500,2000),(1800,2000),(1500,2400),(1800,2400),(1700,2500),(2000,2400)]
    shift = np.asarray(offset)
    return {f'H50Z_030_01_{photo}.bmp': {'keypoints':[
        {'label': f'f{i+1}', 'x_px': float(x+dx), 'y_px':float(y+dy)}
        for i,(x,y) in enumerate(positions)]}
        for photo,(dx,dy) in [('005',(0,0)),('006',tuple(shift))]}


def test_registration_is_2d_geometric_and_no_assumption_about_f8():
    r=fit_common_fiducials(fake_labels())
    assert r['common_fiducial_count']==6
    assert r['max_common_residual_px']<1e-3
    assert np.allclose(r['forward_005_to_006_PRIVATE'][:,2],[-17,29],atol=1e-2)
    assert r['additional_f8_005_to_f7_006_residual_px'] is None


def test_registration_fails_closed_missing_fids():
    labels=fake_labels()
    labels['H50Z_030_01_006.bmp']['keypoints'].pop()
    with pytest.raises(ValueError):fit_common_fiducials(labels)


def test_ridge_support_matches_registered_structure_but_not_truth():
    a=np.full((192,192),210,dtype=np.uint8)
    cv2.line(a,(22,55),(170,75),30,2)
    original=[{'draft_id':'x','points_xy':np.column_stack((np.linspace(22,170,60),
                            np.linspace(55,75,60))).tolist()}]
    same=compare_registered_ridges(a,a.copy(),original)
    blank=np.full_like(a,210)
    blank_case=compare_registered_ridges(a,blank,original)
    assert len(same)==1
    assert (same[0]['counterpart_same_registered_path_coverage'] >
            blank_case[0]['counterpart_same_registered_path_coverage'])
    assert same[0]['native_particle_truth']=='UNKNOWN'
    assert 'displaced_same_shape_counterpart_control_coverage' in same[0]
    assert same[0]['candidate_kept']
    assert not same[0]['physical_auto_join_allowed']


def test_path_support_rejects_bad_inputs():
    a=np.ones((160,160),dtype=np.float32)
    with pytest.raises(ValueError):ridge_near_path(a,np.ones((2,2)))
    with pytest.raises(ValueError):ridge_near_path(a,np.ones((8,2)),radius=7)


def test_exact_uncompressed_bmp_memmap_roi_and_registered_translation(tmp_path):
    from bubble_chamber.s44_crossframe_persistence import ExactBMP8, warp_corresponding_window
    rng = np.random.default_rng(11)
    im = rng.integers(0, 256, size=(1012, 1004), dtype=np.uint8)
    cv2.line(im, (80, 140), (890, 760), 22, 3)
    translation = np.array([[1., 0., 11.], [0., 1., -17.]])
    shifted = cv2.warpAffine(im, translation, (1004,1012),
                              flags=cv2.INTER_NEAREST, borderValue=180)
    pa, pb = tmp_path / 'a.bmp', tmp_path / 'b.bmp'
    assert cv2.imwrite(str(pa), im)
    assert cv2.imwrite(str(pb), shifted)
    a, b = ExactBMP8(pa), ExactBMP8(pb)
    assert np.array_equal(a.crop(152, 310, 128, 160), im[310:470,152:280])
    aligned = warp_corresponding_window(b, (155,311), translation, size=192)
    assert np.array_equal(aligned, a.crop(155,311,192,192))
    with pytest.raises(ValueError):
        a.crop(-1,50,80,90)


def test_bmp_accessor_rejects_bad_layout(tmp_path):
    from bubble_chamber.s44_crossframe_persistence import ExactBMP8
    fake = tmp_path / 'bad.bmp'
    fake.write_bytes(b'BM' + b'\x00' * 52)
    with pytest.raises(ValueError):ExactBMP8(fake)
