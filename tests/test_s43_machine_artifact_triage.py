"""S43 scientific integrity and full-stack feature/graph smoke checks."""
import cv2
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bubble_chamber.s43_artifact_triage import (
    FEATURE_NAMES, resample_polyline, path_features,
    score_native_candidates, annotate_graph_advisory,
    export_simulator_linear_model, infer_simulator_linear_model,
)
from scripts.research.bubble_chamber.s43_machine_artifact_trial import (
    select_calibration_threshold, add_synthetic_scratches, frozen_frame_folds,
    FAMILIES, UNSEEN_FAMILY,
)


def gray():
    arr = np.full((192, 192), 180, dtype='uint8')
    cv2.line(arr, (20, 75), (170, 80), 60, 2)
    return arr


def curve(y=75.):
    return np.column_stack((np.linspace(20, 170, 40),
                            np.linspace(y, y + 5, 40)))


def classifier():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((15, len(FEATURE_NAMES)))
    y = np.arange(len(x)) % 2
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)


def test_path_features_deterministic_and_finite():
    a = gray()
    f = path_features(a, curve())
    g = path_features(a, curve())
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()
    assert np.array_equal(f, g)


def test_bad_paths_are_rejected_not_converted_to_nan():
    a = gray()
    with pytest.raises(ValueError):
        resample_polyline([[3, 4], [3, 4], [3, 4]], a.shape)
    with pytest.raises(ValueError):
        path_features(a, [[10, 10], [30, 30], [400, 11]])
    with pytest.raises(ValueError):
        path_features(a.astype(float), curve())


def test_native_scores_retain_every_candidate_and_never_assign_truth():
    old = [{'draft_id': 'a', 'points_xy': curve().tolist()},
           {'draft_id': 'b', 'points_xy': curve(98).tolist()}]
    original = [dict(c) for c in old]
    scored = score_native_candidates(gray(), old, classifier())
    assert [e['draft_id'] for e in scored] == ['a', 'b']
    assert len(scored) == len(old)
    assert all(e['native_particle_identity'] == 'UNKNOWN' for e in scored)
    assert all(e['candidate_kept'] and not e['can_auto_join_native_tracks'] for e in scored)
    assert old == original
    with pytest.raises(ValueError):
        score_native_candidates(gray(), old + old, classifier())


def test_graph_guard_always_abstains_and_checks_exact_pair_order():
    sc = [{'draft_id': k, 'SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY': v}
          for k, v in [('a', .9), ('b', .1)]]
    graph = {'nodes': [{'source_candidate_ids': ['a']},
                       {'source_candidate_ids': ['b']}],
             'candidate_edges': [{'i': 0, 'j': 1}]}
    s40 = {'nodes': [{}, {}], 'candidate_edges':
           [{'i': 0, 'j': 1, 'fiducial_overlap': True}]}
    result = annotate_graph_advisory(graph, sc, s40)
    assert result['fiducial_risk_edges'] == 1
    assert result['real_physical_auto_joins'] == 0
    assert result['edges'][0]['status'] == 'FIDUCIAL_RISK_ABSTAIN'
    assert result['edges'][0]['ranked_endpoint_similarity_SIMULATOR_ONLY'] == .1
    s40['candidate_edges'][0]['i'] = 1
    with pytest.raises(ValueError):
        annotate_graph_advisory(graph, sc, s40)


def test_nonfiducial_high_score_also_forbids_auto_join():
    sc = [{'draft_id': 'a', 'SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY': .99},
          {'draft_id': 'b', 'SIMULATED_measured_curve_similarity_NOT_NATIVE_PROBABILITY': .99}]
    graph = {'nodes': [{'source_candidate_ids': ['a']},
                       {'source_candidate_ids': ['b']}],
             'candidate_edges': [{'i': 0, 'j': 1}]}
    guard = {'nodes': [{}, {}],
             'candidate_edges': [{'i': 0, 'j': 1, 'fiducial_overlap': False}]}
    result = annotate_graph_advisory(graph, sc, guard)
    assert result['edges'][0]['status'] == 'UNVERIFIED_MACHINE_JOIN_NO_AUTO_ACCEPT'
    assert result['edges'][0]['native_same_particle_relation'] == 'UNKNOWN'
    assert not result['edges'][0]['physical_auto_join_performed']


def test_calibration_threshold_does_not_use_test_labels():
    vals = np.arange(30, dtype=float)
    y = np.zeros(30, dtype=bool)
    y[::3] = True
    assert select_calibration_threshold(vals, y) == np.quantile(vals[y], .90)
    with pytest.raises(ValueError):
        select_calibration_threshold(np.arange(4.), np.array([True, True, False, False]))


def test_scratch_generator_is_reproducible_and_has_open_set_family():
    class Scene:
        def __init__(self):
            self.gray = gray()
            self.right = [curve(), curve(82), curve(100)]
    scene = Scene()
    image, negatives = add_synthetic_scratches(scene, np.random.default_rng(20), open_set=True)
    image2, neg2 = add_synthetic_scratches(scene, np.random.default_rng(20), open_set=True)
    assert np.array_equal(image, image2)
    assert len(negatives) == len(FAMILIES) + 1
    assert [f for f, _ in negatives] == [f for f, _ in neg2]
    assert negatives[-1][0] == UNSEEN_FAMILY
    assert scene.gray[0, 0] == 180  # No mutation of source.


def test_frame_folds_do_not_share_any_measurement_frame():
    names = [f'070_{i:03d}' for i in range(145)]
    from bubble_chamber.s35_simulated_gaps import frame_split
    m = {'frames': [{'frame_id': i} for i in names]}
    split = frozen_frame_folds(m)
    assert [len(split[k]) for k in ['train', 'calibration', 'test']] == [72, 37, 36]
    assert not (split['train'] & split['calibration'])
    assert not (split['train'] & split['test'])
    assert not (split['calibration'] & split['test'])


def test_portable_linear_weights_reproduce_sklearn_and_fail_closed():
    rng=np.random.default_rng(12)
    x=rng.standard_normal((45,len(FEATURE_NAMES)))
    y=np.arange(45)%2
    clf=make_pipeline(StandardScaler(),LogisticRegression(
        class_weight='balanced',max_iter=1000,random_state=5)).fit(x,y)
    model=export_simulator_linear_model(clf,source_photo='005')
    actual=infer_simulator_linear_model(x,model)
    expected=clf.predict_proba(x)[:,1]
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    model['real_physics_autojoin_allowed']=True
    with pytest.raises(ValueError):infer_simulator_linear_model(x,model)
