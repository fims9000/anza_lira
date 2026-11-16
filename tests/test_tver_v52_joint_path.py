"""Minimal deterministic checks for a label-blind V52 research decoder."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tver_v52_joint_path import (
    curvature_ribbon, decode_candidate_v52, infer_joint_path,
)

RECT = (39, 57, 35, 63)


def scene(seed=0, amp=4.0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[:96, :96]
    line = lambda y: 48 + amp * np.sin((y - 38) * np.pi / 19)  # noqa: E731
    ridge = np.exp(-.5 * ((xs - line(ys)) / 1.5) ** 2)
    heat = np.clip(.015 + .92 * ridge + rng.normal(0, .018, (96, 96)), .0001, .999)
    visible = np.clip(.02 + .94 * ridge, 0, 1)
    y0, y1, x0, x1 = RECT
    visible[y0:y1, x0:x1] = 0
    # Curvy competing ridge; not a second oracle truth.
    rival = np.exp(-.5 * ((xs - (48 - amp * np.sin((ys - 38) * np.pi / 19))) / 1.5) ** 2)
    heat = np.clip(np.maximum(heat, .42 * rival), .0001, .999)
    return heat, visible, (31., float(line(31))), (65., float(line(65))), line


def agrees_with_curve(route, line, rect=RECT, tolerance=2):
    y0, y1, x0, x1 = rect
    pred = np.zeros((96, 96), bool)
    for y, x in route:
        if y0 <= y < y1 and x0 <= x < x1:
            pred[y, x] = True
    truth = np.zeros_like(pred)
    for y in range(y0 + 2, y1 - 2):
        x = int(round(line(y)))
        if x0 <= x < x1:
            truth[y, x] = True
    precision = ndi.binary_dilation(truth, iterations=tolerance)[pred].mean() if pred.any() else 0
    recall = ndi.binary_dilation(pred, iterations=tolerance)[truth].mean() if truth.any() else 0
    return precision, recall


def test_decoding_returns_single_raster_connected_curve():
    h, v, a, b, true_curve = scene(3)
    result = infer_joint_path(h, v, RECT, a, b)
    assert result is not None
    assert len(result.vertices) > 15
    assert result.candidates_considered > 0
    steps = np.diff(np.array(result.vertices), axis=0)
    assert np.max(np.abs(steps)) <= 1
    assert agrees_with_curve(result.vertices, true_curve)[1] >= .80


def test_decoder_is_deterministic_given_observations():
    h, v, a, b, _ = scene(17)
    r1 = infer_joint_path(h, v, RECT, a, b)
    r2 = infer_joint_path(h, v, RECT, a, b)
    assert r1 == r2


def test_no_visible_support_abstains_not_straight_hallucination():
    h, v, a, b, _ = scene()
    assert infer_joint_path(h, np.zeros_like(v), RECT, a, b) is None


def test_invalid_probability_range_and_nan_fail_closed():
    h, v, a, b, _ = scene()
    bad = h.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        infer_joint_path(bad, v, RECT, a, b)
    with pytest.raises(ValueError):
        infer_joint_path(h, v + 2, RECT, a, b)
    with pytest.raises(ValueError):
        infer_joint_path(h, v, (39, 57, 35, 100), a, b)
    with pytest.raises(ValueError):
        infer_joint_path(h, v, RECT, a, a)


def test_viterbi_enforces_both_endpoints_even_with_off_center_peak():
    yy, xx = np.mgrid[:96, :96]
    heat = np.exp(-.5 * ((xx - 58) / 2) ** 2)
    route, score = curvature_ribbon(heat, (30, 48), (65, 48), radius=11)
    assert route[0] == (30, 48)
    assert route[-1] == (65, 48)
    assert np.isfinite(score)
    assert all(max(abs(a-c), abs(b-d)) <= 1
               for (a,b),(c,d) in zip(route,route[1:]))


def test_label_blind_public_decoder_does_not_access_expert_fields(monkeypatch):
    h, v, a, b, _ = scene()
    module = types.ModuleType('scripts.tver_v50_conditional_path')
    module.path_heatmap = lambda _model, _scene, _proposal: h
    monkeypatch.setitem(sys.modules, 'scripts.tver_v50_conditional_path', module)
    class ForbiddenFields(dict):
        def __getitem__(self, key):
            assert key not in {'hidden_centerline','true_ends','origin','y','section','kind','label'}
            return super().__getitem__(key)
    obs = ForbiddenFields(visible_prob=v, rectangle=RECT,
                          hidden_centerline='forbidden', section=15, kind='LINK')
    proposal = ForbiddenFields(source=a, target=b, y=1)
    path, meta = decode_candidate_v52(None, obs, proposal)
    assert path is not None
    assert meta['expert_labels_not_used_by_decoder']
    assert meta['mode']=='v52_joint_curvature'


def test_no_correct_class_promise_when_support_missing(monkeypatch):
    h, v, a, b, _ = scene()
    module = types.ModuleType('scripts.tver_v50_conditional_path')
    module.path_heatmap = lambda _model, _scene, _proposal: h
    monkeypatch.setitem(sys.modules, 'scripts.tver_v50_conditional_path', module)
    route, info = decode_candidate_v52(None, {'visible_prob':np.zeros_like(v),
        'rectangle':RECT}, {'source':a,'target':b})
    assert route is None
    assert info['status']=='ABSTAIN_NO_OBSERVED_PAIR'