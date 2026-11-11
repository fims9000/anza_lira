"""S120/S121 synthetic regression; private H50Z data are not required."""
import numpy as np
import cv2
import pytest

from bubble_chamber.s120_photo_features import photo_features, FEATURE_NAMES
from bubble_chamber.s120_review import group_scores
from bubble_chamber.s121_fullframe_continuation import associate_fullframe


def _example(gap_y=80, signal=True):
    image = np.full((200, 240), 215, np.uint8)
    a = np.column_stack((np.arange(20, 80, dtype=float),
                         np.full(60, 80.)))
    b = np.column_stack((np.arange(110, 190, dtype=float),
                         np.full(80, float(gap_y))))
    cv2.line(image, (20, 80), (79, 80), 70, 1)
    cv2.line(image, (110, gap_y), (189, gap_y), 70, 1)
    if signal and gap_y == 80:
        cv2.line(image, (80, 80), (109, 80), 120, 1)
    tiles = [{'tile_id': 'T00000', 'xyxy': [0, 0, 240, 200],
              'curves': [{'candidate_id': 'a', 'observed_pixel_path_xy': a.tolist()},
                         {'candidate_id': 'b', 'observed_pixel_path_xy': b.tolist()}]}]
    groups = [{'global_group_id': 'ga', 'source_curve_keys': ['T00000:a']},
              {'global_group_id': 'gb', 'source_curve_keys': ['T00000:b']}]
    return image, tiles, groups


def test_source_photo_features_reverse_invariant():
    image, tiles, _ = _example()
    xy = np.asarray(tiles[0]['curves'][0]['observed_pixel_path_xy'])
    forward, backward = photo_features(image, xy), photo_features(image, xy[::-1])
    assert len(forward) == len(FEATURE_NAMES) == len(backward)
    assert np.all(np.isfinite(forward))
    np.testing.assert_allclose(forward, backward, atol=.15)


def test_features_use_actual_source_image():
    image, tiles, _ = _example()
    pts = tiles[0]['curves'][0]['observed_pixel_path_xy']
    actual = photo_features(image, pts)
    blank = photo_features(np.full_like(image, 215), pts)
    ix = FEATURE_NAMES.index('mean_flank_contrast_8')
    assert actual[ix] > blank[ix] + 20.


def test_one_color_does_not_destroy_separate_group_ids():
    groups = [{'global_group_id': 'same', 'source_curve_keys': ['a', 'b']},
              {'global_group_id': 'other', 'source_curve_keys': ['c']}]
    out = group_scores(groups, {'a': .95, 'b': .6, 'c': .1},
                       {'a': 100., 'b': 80., 'c': 10.})
    assert out['same']['decision'] == 'AUTO_TRACK_LIKE_NOT_GT'
    assert out['other']['decision'] == 'DEPRIORITIZED_NOT_DELETED'
    assert [g['global_group_id'] for g in groups] == ['same', 'other']


def test_s121_clean_photo_supported_gap_is_provisional_only():
    gray, tiles, groups = _example()
    out = associate_fullframe(tiles, {}, groups, gray)
    assert out['report']['accepted_provisional_links'] == 1
    assert len(out['groups']) == 1
    link = next(x for x in out['proposals'] if x['decision'].startswith('AUTO_'))
    assert link['source_observed_bridge_pixels'] == 0
    assert link['inferred_geometry_only']


def test_s121_blank_gap_is_not_accepted():
    gray, tiles, groups = _example(signal=False)
    out = associate_fullframe(tiles, {}, groups, gray)
    assert out['report']['accepted_provisional_links'] == 0
    assert len(out['groups']) == 2


def test_s121_parallel_jump_rejected():
    gray, tiles, groups = _example(gap_y=86)
    out = associate_fullframe(tiles, {}, groups, gray)
    assert out['report']['accepted_provisional_links'] == 0


def test_s121_already_same_group_does_not_close_a_cycle():
    gray, tiles, _ = _example()
    groups = [{'global_group_id': 'ga',
               'source_curve_keys': ['T00000:a', 'T00000:b']}]
    out = associate_fullframe(tiles, {}, groups, gray)
    assert len(out['groups']) == 1
    assert out['report']['accepted_provisional_links'] == 0


def test_s121_rejects_wrong_source_image_or_outside_points():
    gray, tiles, groups = _example()
    with pytest.raises(ValueError, match='grayscale'):
        associate_fullframe(tiles, {}, groups, gray.astype(np.float32))
    tiles[0]['curves'][0]['observed_pixel_path_xy'][0] = [-5., 80.]
    with pytest.raises(ValueError, match='outside'):
        associate_fullframe(tiles, {}, groups, gray)
