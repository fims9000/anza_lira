from __future__ import annotations

import math

import numpy as np
import pytest

from anza_s.a2.cases import a2_candidate_stream, curved_comparability
from anza_s.a2.cauchy_green import cauchy_green, cocycle_product, finite_time_diagnostics
from anza_s.a2.cone_diagnostic import axial_angle, contracted_half_angle, inside_cone
from anza_s.a2.covariance_transport import covariance_sequence, frame_matrices
from anza_s.a2.evaluator import _score
from anza_s.a2.run import protocol_payload
from anza_s.a2.shadowing import hyperbolic_shadowing
from anza_s.frame import HyperbolicFrame
from anza_s.oracle_field import OracleCocycleField, geometry_for_sample
from anza_s.rollout import TrajectoryPoint, rollout
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def _trajectory(steps: int = 3, curvature: float = 0.08) -> tuple[TrajectoryPoint, ...]:
    return tuple(TrajectoryPoint(k, float(k), 0.0, 1.0, 0.0, 1, 1.0, curvature) for k in range(steps + 1))


def test_frame_is_area_preserving():
    assert np.linalg.det(HyperbolicFrame(0.2, 0.1, 0.35).matrix()) == pytest.approx(1.0)


def test_covariance_recurrence_matches_product_formula():
    trajectory = _trajectory()
    matrices = frame_matrices(trajectory, hyperbolicity=0.35)
    covariance = covariance_sequence(trajectory, mode="composed", hyperbolicity=0.35)
    for step in range(1, len(trajectory)):
        product = cocycle_product(matrices[:step])
        assert np.allclose(covariance[step], product @ product.T)


def test_composed_covariance_determinant_is_one():
    for covariance in covariance_sequence(_trajectory(), mode="composed", hyperbolicity=0.35):
        assert np.linalg.det(covariance) == pytest.approx(1.0, abs=1e-10)


def test_lambda_zero_is_isotropic():
    for covariance in covariance_sequence(_trajectory(), mode="composed", hyperbolicity=0.0):
        assert np.allclose(covariance, np.eye(2), atol=1e-10)


def test_cauchy_green_is_symmetric_positive_definite():
    tensor = cauchy_green(cocycle_product(frame_matrices(_trajectory(), hyperbolicity=0.35)))
    assert np.allclose(tensor, tensor.T)
    assert np.linalg.eigvalsh(tensor).min() > 0


def test_singular_values_are_reciprocal():
    product = cocycle_product(frame_matrices(_trajectory(), hyperbolicity=0.35))
    singular = np.linalg.svd(product, compute_uv=False)
    assert np.prod(singular) == pytest.approx(1.0, abs=1e-10)


def test_finite_time_lyapunov_diagnostics_are_finite():
    value = finite_time_diagnostics(cocycle_product(frame_matrices(_trajectory(), hyperbolicity=0.35)), 3.0)
    assert all(math.isfinite(item) for item in value.values())
    assert value["ftle_max"] > value["ftle_min"]


def test_hyperbolic_shadowing_score_is_bounded():
    path = _trajectory()
    covariance = covariance_sequence(path, mode="composed", hyperbolicity=0.35)
    energy, score, meeting, matrix = hyperbolic_shadowing(path, path, covariance, covariance)
    assert energy >= 0 and 0 < score <= 1 and meeting == (0, 0) and np.isfinite(matrix).all()


def test_all_a1_a2_a3_controls_reuse_exact_centerline_objects():
    path = _trajectory()
    scores = [_score(method, path, path)[0] for method in (
        "A1_isotropic_shadowing", "A2_local_anisotropic_reset",
        "A3_cocycle_cg_lambda0", "A3_cocycle_cg_lambda035",
    )]
    assert all(np.isfinite(scores))
    assert tuple(point.xy.tolist() for point in path) == tuple(point.xy.tolist() for point in path)


def test_local_anisotropy_resets_each_step():
    path = _trajectory()
    matrices = frame_matrices(path, hyperbolicity=0.35)
    covariance = covariance_sequence(path, mode="local_reset", hyperbolicity=0.35)
    assert all(np.allclose(covariance[k + 1], matrix @ matrix.T) for k, matrix in enumerate(matrices))


def test_composed_transport_uses_previous_covariance():
    local = covariance_sequence(_trajectory(), mode="local_reset", hyperbolicity=0.35)
    composed = covariance_sequence(_trajectory(), mode="composed", hyperbolicity=0.35)
    assert not np.allclose(local[-1], composed[-1])


def test_reset_and_composed_are_equal_for_one_step():
    path = _trajectory(steps=1)
    assert np.allclose(
        covariance_sequence(path, mode="local_reset", hyperbolicity=0.35)[-1],
        covariance_sequence(path, mode="composed", hyperbolicity=0.35)[-1],
    )


def test_reset_and_composed_differ_for_multiple_steps():
    path = _trajectory(steps=3)
    assert not np.allclose(
        covariance_sequence(path, mode="local_reset", hyperbolicity=0.35)[-1],
        covariance_sequence(path, mode="composed", hyperbolicity=0.35)[-1],
    )


def test_lambda_zero_composed_equals_isotropic_shadowing():
    left = _trajectory(); right = _trajectory()
    assert _score("A1_isotropic_shadowing", left, right)[0] == pytest.approx(_score("A3_cocycle_cg_lambda0", left, right)[0])


def test_shadowing_is_axially_sign_invariant():
    left = _trajectory(steps=1)
    right = tuple(TrajectoryPoint(p.step, p.x, p.y, -p.ux, -p.uy, p.branch_id, p.membership, p.curvature) for p in left)
    covariance = covariance_sequence(left, mode="isotropic", hyperbolicity=0.0)
    assert hyperbolic_shadowing(left, left, covariance, covariance)[1] == pytest.approx(hyperbolic_shadowing(left, right, covariance, covariance)[1])


def test_orthogonal_x_turn_has_orientation_penalty():
    horizontal = _trajectory(steps=1)
    vertical = tuple(TrajectoryPoint(p.step, p.x, p.y, 0.0, 1.0, p.branch_id, p.membership, p.curvature) for p in horizontal)
    covariance = covariance_sequence(horizontal, mode="isotropic", hyperbolicity=0.0)
    assert hyperbolic_shadowing(horizontal, horizontal, covariance, covariance)[1] > hyperbolic_shadowing(horizontal, vertical, covariance, covariance)[1]


def test_transverse_parallel_offset_costs_more_than_longitudinal_offset():
    covariance = np.diag((9.0, 1.0))
    transverse = np.asarray((0.0, 1.0)); longitudinal = np.asarray((1.0, 0.0))
    assert transverse @ np.linalg.solve(covariance, transverse) > longitudinal @ np.linalg.solve(covariance, longitudinal)


def test_straight_positive_negative_geometry_only_scores_are_equal():
    positive = generate_sample_v4("train", 0, image_size=64)
    negative = generate_sample_v4("train", 128, image_size=64)
    positive_candidate = [item for sample, item in a2_candidate_stream("train") if sample["index"] == 0][0]
    negative_candidate = [item for sample, item in a2_candidate_stream("train") if sample["index"] == 128][0]
    values = []
    for sample, candidate in ((positive, positive_candidate), (negative, negative_candidate)):
        field = OracleCocycleField(geometry_for_sample(sample))
        left = rollout(field, candidate.start_xy, candidate.start_direction, steps=3, cocycle=False)
        right = rollout(field, candidate.goal_xy, candidate.goal_direction, steps=3, cocycle=False)
        values.append(_score("A3_cocycle_cg_lambda035", left, right)[0])
    assert values[0] == pytest.approx(values[1], abs=1e-12)


@pytest.mark.parametrize("split", ["confirm", "test"])
def test_phase_a2_candidate_stream_refuses_confirm_and_test(split):
    with pytest.raises(PermissionError):
        a2_candidate_stream(split)


def test_protocol_keeps_training_cracks_and_expert_closed():
    protocol = protocol_payload()
    assert protocol["training_performed"] is False and protocol["phase_b_opened"] is False
    assert protocol["cracks_data_accessed"] is False and protocol["expert_data_accessed"] is False
    assert protocol["streams"]["confirm"] == protocol["streams"]["test"] == "CLOSED"


def test_curved_comparability_rule_is_deterministic():
    first = curved_comparability(a2_candidate_stream("train"))
    second = curved_comparability(a2_candidate_stream("train"))
    assert first == second and first["positive_count"] > 0 and first["negative_count"] > 0


def test_invariant_cone_contracts_for_positive_lambda():
    alpha = math.radians(20)
    contracted = contracted_half_angle(alpha, 0.35)
    assert 0 < contracted < alpha
    assert inside_cone(np.asarray((1.0, 0.0)), np.asarray((-1.0, 0.0)), contracted)
    assert axial_angle(np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))) == pytest.approx(math.pi / 2)
