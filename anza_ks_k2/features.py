"""K1-compatible information features without mutating frozen ``anza_ks``."""

from __future__ import annotations

import numpy as np

from anza_ks.constants import FEATURE_WIDTH
from anza_ks.entropy import block_entropy, conditional_entropies
from anza_ks.features import _coarse_masses, itineraries_for, raw_dynamic_features
from anza_ks.partitions import PARTITION_COUNT
from anza_ks.predictive_info import predictive_information
from anza_ks.symbolic_mass import image_density, symbolic_probabilities
from anza_ks.torus import CAT_MAP, SHEAR_MAP


def information_features(patch: np.ndarray, dynamics: str) -> np.ndarray:
    """Return the exact frozen K1 information family for Cat or shear dynamics."""

    if dynamics not in {"cat", "shear"}:
        raise ValueError("dynamics must be 'cat' or 'shear'")
    matrix = CAT_MAP if dynamics == "cat" else SHEAR_MAP
    array = np.asarray(patch, dtype=np.float64)
    itinerary = itineraries_for(dynamics, array.shape[0])
    density = image_density(array)
    forward_entropy: list[float] = []
    backward_entropy: list[float] = []
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
    _, coarse_entropy = _coarse_masses(density, itinerary, matrix)
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


def shear_ks_feature_vector(patch: np.ndarray) -> np.ndarray:
    raw = raw_dynamic_features(np.asarray(patch), "shear")
    information = information_features(np.asarray(patch), "shear")
    values = np.concatenate((raw, information))
    if len(values) > FEATURE_WIDTH:
        raise ValueError("ShearKS exceeds frozen K1 feature width")
    return np.pad(values, (0, FEATURE_WIDTH - len(values)))
