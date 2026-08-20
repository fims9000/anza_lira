"""K0 exact mathematics and finite-information sanity audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..entropy import block_entropy
from ..itineraries import precompute_itineraries
from ..orientation_bank import align_patch
from ..predictive_info import predictive_information
from ..symbolic_mass import image_density, symbolic_probabilities
from ..torus import CAT_INVERSE, CAT_MAP, exact_permutation


def run_k0_math() -> dict[str, Any]:
    size = 17
    rng = np.random.default_rng(202_608_19)
    field = rng.integers(-100, 101, size=(size, size), dtype=np.int64)
    permuted = exact_permutation(field, CAT_MAP)
    recovered = exact_permutation(permuted, CAT_INVERSE)
    itinerary = precompute_itineraries(size=size, K=4)
    uniform = np.full((size, size), 1.0 / (size * size))
    delta = np.zeros((size, size))
    delta[size // 2, size // 2] = 1.0
    uniform_prob = symbolic_probabilities(uniform, itinerary.forward_word_ids[4], 4**4)
    delta_prob = symbolic_probabilities(delta, itinerary.forward_word_ids[4], 4**4)
    patch = rng.normal(size=(size, size))
    density = image_density(patch)
    scaled_density = image_density(4.3 * patch - 2.1)
    predictive = predictive_information(
        density,
        itinerary.predictive_past_ids,
        itinerary.predictive_future_ids,
        itinerary.predictive_joint_ids,
        alphabet_size=16,
    )
    checks = {
        "cat_determinant_one": int(round(np.linalg.det(CAT_MAP))) == 1,
        "cat_inverse_exact": np.array_equal(CAT_MAP @ CAT_INVERSE, np.eye(2, dtype=np.int64)),
        "finite_permutation_inverse_exact": np.array_equal(recovered, field),
        "finite_permutation_mass_exact": float(permuted.sum()) == float(field.sum()),
        "finite_permutation_l2_exact": bool(np.isclose(np.linalg.norm(permuted), np.linalg.norm(field), atol=1e-12)),
        "word_ids_deterministic": np.array_equal(itinerary.forward_word_ids[4], precompute_itineraries(size=size, K=4).forward_word_ids[4]),
        "density_sums_one": bool(np.isclose(density.sum(), 1.0)),
        "entropy_finite_nonnegative": bool(np.isfinite(block_entropy(uniform_prob)) and block_entropy(uniform_prob) >= 0),
        "delta_more_concentrated": block_entropy(delta_prob) < block_entropy(uniform_prob),
        "predictive_information_nonnegative": bool(np.isfinite(predictive) and predictive >= -1e-12),
        "contrast_density_invariance": bool(np.allclose(density, scaled_density, atol=1e-9)),
        "axial_chart_periodicity": bool(np.allclose(align_patch(patch, 0.31), align_patch(patch, 0.31 + np.pi))),
        "finite_partition_not_markov_claim": True,
        "no_exact_image_ks_entropy_claim": True,
    }
    return {
        "status": "ANZA_KS_K0_MATH_PASS" if all(checks.values()) else "ANZA_KS_K0_MATH_FAIL",
        "checks": checks,
        "uniform_H4": block_entropy(uniform_prob),
        "delta_H4": block_entropy(delta_prob),
        "predictive_information_fixture": predictive,
        "training_performed": False,
    }
