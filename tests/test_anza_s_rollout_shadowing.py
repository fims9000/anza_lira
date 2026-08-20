import numpy as np

from anza_s.frame import HyperbolicFrame
from anza_s.oracle_field import OracleCocycleField
from anza_s.rollout import (
    TrajectoryPoint, bilinear_sample, cocycle_product, residual_output, rollout, stable_widths,
)
from anza_s.shadowing import meeting_energy_matrix, two_sided_shadowing
from synthetic.geometry_generator import generate_geometry, scale_geometry


def _geometry(case: str, seed: int = 7):
    return scale_geometry(generate_geometry(case, np.random.default_rng(seed)), 64)


def _point(step, xy, direction):
    unit = np.asarray(direction, dtype=float); unit /= np.linalg.norm(unit)
    return TrajectoryPoint(step, xy[0], xy[1], unit[0], unit[1], 1, 1.0, 0.0)


def _trajectory(points, directions):
    return tuple(_point(i, xy, direction) for i, (xy, direction) in enumerate(zip(points, directions, strict=True)))


def test_09_continuous_frame_does_not_turn_at_x():
    geometry = _geometry("x_junction")
    field = OracleCocycleField(geometry)
    first, correct, wrong = geometry.branches[0], geometry.branches[1], geometry.branches[3]
    start = first.points_xy[-6]; direction = first.points_xy[-1] - start
    path = rollout(field, start, direction, steps=8, cocycle=True)
    terminal = path[-1].xy
    correct_distance = np.linalg.norm(correct.points_xy - terminal, axis=1).min()
    wrong_distance = np.linalg.norm(wrong.points_xy - terminal, axis=1).min()
    assert correct_distance < wrong_distance


def test_10_cocycle_tracks_a_curved_trace():
    geometry = _geometry("curved_fault")
    branch = geometry.branches[0]; field = OracleCocycleField(geometry)
    start = branch.points_xy[30]; direction = branch.points_xy[31] - start
    path = rollout(field, start, direction, steps=10, cocycle=True)
    distance = np.linalg.norm(branch.points_xy - path[-1].xy, axis=1).min()
    assert distance < 1.5


def test_11_forward_backward_rollouts_meet_on_straight_trace():
    geometry = _geometry("single_straight")
    branch = geometry.branches[0]; field = OracleCocycleField(geometry)
    left, right = branch.points_xy[38], branch.points_xy[56]
    forward = rollout(field, left, right - left, steps=6, cocycle=True)
    backward = rollout(field, right, left - right, steps=6, cocycle=True)
    energy, score, _meeting = two_sided_shadowing(forward, backward)
    assert energy < 1.5
    assert score > 0.2


def test_12_bilinear_sampling_is_finite():
    image = np.arange(16, dtype=float).reshape(4, 4)
    value = bilinear_sample(image, np.asarray((1.25, 2.5)))
    assert np.isfinite(value)
    assert np.isclose(value, 11.25)


def test_13_cocycle_composition_matches_sequential_matrices():
    frames = (HyperbolicFrame(0.1, 0.05, 0.2), HyperbolicFrame(0.15, -0.02, 0.3))
    vector = np.asarray((0.7, 0.2))
    sequential = frames[1].matrix() @ (frames[0].matrix() @ vector)
    assert np.allclose(cocycle_product(frames) @ vector, sequential)


def test_14_stable_width_decreases():
    widths = stable_widths(6, 0.35)
    assert np.all(np.diff(widths) < 0)
    assert widths[-1] > 0.25


def test_15_zero_residual_scale_is_identity():
    base = np.arange(9, dtype=float).reshape(3, 3)
    assert np.array_equal(residual_output(base, np.ones_like(base), gamma=0.0), base)


def test_16_correct_straight_pair_has_lower_energy_than_wrong_orientation():
    left = _trajectory([(0, 0), (1, 0), (2, 0)], [(1, 0)] * 3)
    correct = _trajectory([(4, 0), (3, 0), (2.1, 0)], [(-1, 0)] * 3)
    wrong = _trajectory([(4, 0), (3, 0), (2.1, 0)], [(0, 1)] * 3)
    assert two_sided_shadowing(left, correct)[0] < two_sided_shadowing(left, wrong)[0]


def test_17_correct_curved_pair_has_lower_energy():
    left = _trajectory([(0, 2), (1, 1), (2, 0.4)], [(1, -1), (1, -0.7), (1, -0.3)])
    correct = _trajectory([(4, 2), (3, 1), (2.2, 0.4)], [(-1, -1), (-1, -0.7), (-1, -0.3)])
    wrong = _trajectory([(4, 4), (3, 4), (2.2, 4)], [(-1, 0)] * 3)
    assert two_sided_shadowing(left, correct)[0] < two_sided_shadowing(left, wrong)[0]


def test_18_x_wrong_turn_is_penalized_by_orientation():
    horizontal = _trajectory([(-2, 0), (-1, 0), (0, 0)], [(1, 0)] * 3)
    continuation = _trajectory([(2, 0), (1, 0), (0, 0)], [(-1, 0)] * 3)
    vertical = _trajectory([(0, 2), (0, 1), (0, 0)], [(0, -1)] * 3)
    assert meeting_energy_matrix(horizontal, continuation).min() < meeting_energy_matrix(horizontal, vertical).min()


def test_19_parallel_pair_is_penalized_by_spatial_separation():
    left = _trajectory([(0, 0), (1, 0), (2, 0)], [(1, 0)] * 3)
    same = _trajectory([(4, 0), (3, 0), (2.1, 0)], [(-1, 0)] * 3)
    parallel = _trajectory([(4, 3), (3, 3), (2.1, 3)], [(-1, 0)] * 3)
    assert two_sided_shadowing(left, same)[0] < two_sided_shadowing(left, parallel)[0]


def test_20_observable_negative_gap_has_lower_score():
    left = _trajectory([(0, 0), (1, 0), (2, 0)], [(1, 0)] * 3)
    positive = _trajectory([(4, 0), (3, 0), (2.1, 0)], [(-1, 0)] * 3)
    negative = _trajectory([(4, 3), (3, 3), (2.1, 3)], [(-1, 0)] * 3)
    assert two_sided_shadowing(left, positive)[1] > two_sided_shadowing(left, negative)[1]


def test_21_matched_gap_lineage_alone_is_not_observable_to_shadowing():
    positive = _geometry("fault_with_gap", seed=11)
    negative = _geometry("negative_gap", seed=11)
    positive_field, negative_field = OracleCocycleField(positive), OracleCocycleField(negative)
    p_start, p_goal = positive.gaps[0].points_xy[[0, -1]]
    n_start, n_goal = negative.gaps[0].points_xy[[0, -1]]
    p_score = two_sided_shadowing(
        rollout(positive_field, p_start, p_goal - p_start, steps=4, cocycle=True),
        rollout(positive_field, p_goal, p_start - p_goal, steps=4, cocycle=True),
    )[1]
    n_score = two_sided_shadowing(
        rollout(negative_field, n_start, n_goal - n_start, steps=4, cocycle=True),
        rollout(negative_field, n_goal, n_start - n_goal, steps=4, cocycle=True),
    )[1]
    assert np.isclose(p_score, n_score, atol=1e-12)
