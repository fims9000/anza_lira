import numpy as np
import torch

from anza_ks.block import ANZAKSResidualReadout
from anza_ks.entropy import block_entropy
from anza_ks.constants import FEATURE_WIDTH
from anza_ks.features import dynamic_feature_vector, kolmogorov_information_features
from anza_ks.itineraries import encode_word, precompute_itineraries
from anza_ks.orientation_bank import align_patch
from anza_ks.predictive_info import predictive_information
from anza_ks.symbolic_mass import image_density, symbolic_probabilities
from anza_ks.torus import CAT_INVERSE, CAT_MAP, exact_permutation


def test_cat_permutation_inverse_and_mass_exact():
    field = np.arange(17 * 17).reshape(17, 17)
    forward = exact_permutation(field, CAT_MAP)
    recovered = exact_permutation(forward, CAT_INVERSE)
    assert np.array_equal(recovered, field)
    assert forward.sum() == field.sum()
    assert np.linalg.norm(forward) == np.linalg.norm(field)


def test_symbolic_words_are_deterministic_and_forward_backward_correct():
    first = precompute_itineraries(size=17, K=4)
    second = precompute_itineraries(size=17, K=4)
    assert np.array_equal(first.forward_word_ids[4], second.forward_word_ids[4])
    manual_forward = encode_word([first.symbols_by_lag[k] for k in (0, 1, 2)])
    manual_backward = encode_word([first.symbols_by_lag[k] for k in (0, -1, -2)])
    assert np.array_equal(first.forward_word_ids[3], manual_forward)
    assert np.array_equal(first.backward_word_ids[3], manual_backward)


def test_density_and_entropy_sanity_and_contrast_invariance():
    rng = np.random.default_rng(9)
    patch = rng.normal(size=(17, 17))
    density = image_density(patch)
    scaled = image_density(3.7 * patch + 4.2)
    assert np.isclose(density.sum(), 1.0)
    assert np.all(density > 0)
    assert np.allclose(density, scaled, atol=1e-9)
    itinerary = precompute_itineraries(size=17, K=4)
    probability = symbolic_probabilities(density, itinerary.forward_word_ids[4], 4**4)
    assert np.isclose(probability.sum(), 1.0)
    assert np.isfinite(block_entropy(probability)) and block_entropy(probability) >= 0


def test_delta_density_is_more_concentrated_than_uniform_for_single_symbols():
    itinerary = precompute_itineraries(size=17, K=4)
    uniform = np.full((17, 17), 1.0 / (17 * 17))
    delta = np.zeros((17, 17))
    delta[8, 8] = 1.0
    uniform_p = symbolic_probabilities(uniform, itinerary.forward_word_ids[1], 4)
    delta_p = symbolic_probabilities(delta, itinerary.forward_word_ids[1], 4)
    assert block_entropy(delta_p) < block_entropy(uniform_p)


def test_predictive_information_nonnegative():
    itinerary = precompute_itineraries(size=17, K=4)
    density = image_density(np.eye(17))
    value = predictive_information(
        density,
        itinerary.predictive_past_ids,
        itinerary.predictive_future_ids,
        itinerary.predictive_joint_ids,
        alphabet_size=16,
    )
    assert np.isfinite(value) and value >= -1e-12


def test_orientation_chart_is_axially_periodic():
    patch = np.arange(17 * 17, dtype=np.float64).reshape(17, 17)
    assert np.allclose(align_patch(patch, 0.37), align_patch(patch, 0.37 + np.pi))


def test_feature_widths_are_capacity_aligned_and_finite():
    patch = np.random.default_rng(4).normal(size=(17, 17))
    for method in ("K1_B_shear_raw", "K1_C_cat_raw", "K1_D_anza_ks"):
        vector = dynamic_feature_vector(patch, method)
        assert vector.shape == (FEATURE_WIDTH,)
        assert np.isfinite(vector).all()
    assert len(kolmogorov_information_features(patch)) == 25


def test_gamma_zero_is_exact_identity_and_gradients_are_finite():
    module = ANZAKSResidualReadout(channels=3, feature_width=7, orientation_count=8)
    x = torch.randn(2, 5, 3, requires_grad=True)
    symbolic = torch.randn(2, 5, 8, 7, requires_grad=True)
    evidence = torch.randn(2, 5, 8, requires_grad=True)
    output = module(x, symbolic, evidence)
    assert torch.equal(output, x)
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
