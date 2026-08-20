import math

import numpy as np

from anza_s.frame import HyperbolicFrame, axial_compatibility, match_transported_frame


def test_01_cocycle_determinant_is_one():
    frame = HyperbolicFrame(0.4, 0.12, 0.7)
    assert np.isclose(np.linalg.det(frame.matrix()), 1.0, atol=1e-12)


def test_02_axial_sign_invariance():
    axis = np.asarray((0.6, 0.8))
    assert np.isclose(axial_compatibility(axis, axis), axial_compatibility(axis, -axis))


def test_03_tangent_maps_to_outgoing_tangent():
    frame = HyperbolicFrame(0.3, -0.17, 0.5)
    assert np.allclose(frame.transport(frame.tangent), frame.outgoing_tangent, atol=1e-12)


def test_04_normal_maps_to_outgoing_normal():
    frame = HyperbolicFrame(0.3, -0.17, 0.5)
    assert np.allclose(frame.transport(frame.normal), frame.outgoing_normal, atol=1e-12)


def test_05_tangent_expansion_is_exponential():
    frame = HyperbolicFrame(-0.2, 0.08, 0.6)
    assert np.isclose(np.linalg.norm(frame.transport(frame.tangent, normalize=False)), math.exp(0.6))


def test_06_normal_contraction_is_exponential():
    frame = HyperbolicFrame(-0.2, 0.08, 0.6)
    assert np.isclose(np.linalg.norm(frame.transport(frame.normal, normalize=False)), math.exp(-0.6))


def test_07_inverse_transport_is_exact():
    frame = HyperbolicFrame(0.7, 0.13, 0.4)
    vector = np.asarray((0.2, 0.98))
    restored = frame.inverse_transport(frame.transport(vector, normalize=False), normalize=False)
    assert np.allclose(restored, vector, atol=1e-12)


def test_08_soft_mode_matching_is_permutation_invariant():
    axes = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    membership = np.asarray((0.9, 0.7, 0.2))
    reference, weights = match_transported_frame(np.asarray((1.0, 0.1)), axes, membership)
    permutation = np.asarray((2, 0, 1))
    permuted, permuted_weights = match_transported_frame(
        np.asarray((1.0, 0.1)), axes[permutation], membership[permutation]
    )
    assert np.allclose(reference, permuted)
    assert np.allclose(weights[permutation], permuted_weights)
