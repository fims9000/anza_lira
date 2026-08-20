"""Finite past--future predictive information."""

from __future__ import annotations

import numpy as np

from .entropy import block_entropy
from .symbolic_mass import symbolic_probabilities


def predictive_information(
    density: np.ndarray,
    past_ids: np.ndarray,
    future_ids: np.ndarray,
    joint_ids: np.ndarray,
    *,
    alphabet_size: int = 16,
) -> float:
    past = symbolic_probabilities(density, past_ids, alphabet_size)
    future = symbolic_probabilities(density, future_ids, alphabet_size)
    joint = symbolic_probabilities(density, joint_ids, alphabet_size * alphabet_size)
    value = block_entropy(past) + block_entropy(future) - block_entropy(joint)
    return float(max(value, 0.0))
