"""Capacity-aligned K1 feature vectors."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .entropy import block_entropy, conditional_entropies
from .constants import FEATURE_WIDTH
from .itineraries import SymbolicItineraries, precompute_itineraries
from .koopman_probes import koopman_correlations
from .partitions import PARTITION_COUNT
from .predictive_info import predictive_information
from .symbolic_mass import image_density, symbolic_probabilities
from .torus import CAT_MAP, SHEAR_MAP, exact_permutation


METHODS = ("K1_A_static", "K1_B_shear_raw", "K1_C_cat_raw", "K1_D_anza_ks")


@lru_cache(maxsize=4)
def itineraries_for(method: str, size: int = 17) -> SymbolicItineraries:
    if method == "shear":
        return precompute_itineraries(size=size, K=4, matrix=SHEAR_MAP)
    if method == "cat":
        return precompute_itineraries(size=size, K=4, matrix=CAT_MAP)
    raise ValueError(f"unknown dynamics {method}")


def _pad(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if len(values) > FEATURE_WIDTH:
        raise ValueError(f"feature vector exceeds frozen width {FEATURE_WIDTH}: {len(values)}")
    return np.pad(values, (0, FEATURE_WIDTH - len(values)))


def _coarse_masses(density: np.ndarray, itinerary: SymbolicItineraries, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    partition = itinerary.symbols_by_lag[0]
    masses = []
    entropies = []
    for lag in range(-itinerary.K, itinerary.K + 1):
        transported = exact_permutation(density, matrix, power=-lag)
        probability = symbolic_probabilities(transported, partition, PARTITION_COUNT)
        masses.extend(probability)
        entropies.append(block_entropy(probability))
    return np.asarray(masses), np.asarray(entropies)


def raw_dynamic_features(patch: np.ndarray, dynamics: str) -> np.ndarray:
    matrix = SHEAR_MAP if dynamics == "shear" else CAT_MAP
    itinerary = itineraries_for(dynamics, np.asarray(patch).shape[0])
    density = image_density(patch)
    masses, _ = _coarse_masses(density, itinerary, matrix)
    correlations = koopman_correlations(patch, matrix, K=4)
    return np.concatenate((masses, correlations))


def kolmogorov_information_features(patch: np.ndarray) -> np.ndarray:
    itinerary = itineraries_for("cat", np.asarray(patch).shape[0])
    density = image_density(patch)
    forward_entropy = []
    backward_entropy = []
    for length in range(1, itinerary.K + 1):
        count = PARTITION_COUNT**length
        forward_entropy.append(block_entropy(symbolic_probabilities(density, itinerary.forward_word_ids[length], count)))
        backward_entropy.append(block_entropy(symbolic_probabilities(density, itinerary.backward_word_ids[length], count)))
    forward_conditional = conditional_entropies(forward_entropy)
    backward_conditional = conditional_entropies(backward_entropy)
    predictive = predictive_information(
        density,
        itinerary.predictive_past_ids,
        itinerary.predictive_future_ids,
        itinerary.predictive_joint_ids,
        alphabet_size=16,
    )
    asymmetry_h = float(np.mean(np.abs(np.asarray(forward_entropy) - np.asarray(backward_entropy))))
    asymmetry_conditional = float(np.mean(np.abs(forward_conditional - backward_conditional)))
    _, coarse_entropy = _coarse_masses(density, itinerary, CAT_MAP)
    entropy_production = np.diff(coarse_entropy)
    return np.concatenate(
        (
            np.asarray(forward_entropy),
            np.asarray(backward_entropy),
            forward_conditional,
            backward_conditional,
            [predictive, asymmetry_h, asymmetry_conditional],
            entropy_production,
        )
    )


def dynamic_feature_vector(patch: np.ndarray, method: str) -> np.ndarray:
    if method == "K1_B_shear_raw":
        return _pad(raw_dynamic_features(patch, "shear"))
    if method == "K1_C_cat_raw":
        return _pad(raw_dynamic_features(patch, "cat"))
    if method == "K1_D_anza_ks":
        raw = raw_dynamic_features(patch, "cat")
        information = kolmogorov_information_features(patch)
        return _pad(np.concatenate((raw, information)))
    raise ValueError(f"not a dynamic method: {method}")
