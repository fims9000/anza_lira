"""Precomputed finite symbolic words under exact torus permutations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .partitions import PARTITION_COUNT, quadrant_partition
from .torus import CAT_MAP, exact_permutation


@dataclass(frozen=True)
class SymbolicItineraries:
    size: int
    K: int
    symbols_by_lag: dict[int, np.ndarray]
    forward_word_ids: dict[int, np.ndarray]
    backward_word_ids: dict[int, np.ndarray]
    predictive_joint_ids: np.ndarray
    predictive_past_ids: np.ndarray
    predictive_future_ids: np.ndarray


def encode_word(symbols: list[np.ndarray]) -> np.ndarray:
    if not symbols:
        raise ValueError("a symbolic word needs at least one symbol")
    result = np.zeros_like(symbols[0], dtype=np.int64)
    multiplier = 1
    for symbol in symbols:
        result += np.asarray(symbol, dtype=np.int64) * multiplier
        multiplier *= PARTITION_COUNT
    return result


def precompute_itineraries(
    *, size: int = 17, K: int = 4, matrix: np.ndarray = CAT_MAP, predictive_length: int = 2
) -> SymbolicItineraries:
    if K < 1 or predictive_length < 1 or predictive_length > K:
        raise ValueError("require 1 <= predictive_length <= K")
    partition = quadrant_partition(size)
    symbols = {lag: exact_permutation(partition, matrix, power=lag) for lag in range(-K, K + 1)}
    forward = {length: encode_word([symbols[lag] for lag in range(length)]) for length in range(1, K + 1)}
    backward = {length: encode_word([symbols[-lag] for lag in range(length)]) for length in range(1, K + 1)}
    past = encode_word([symbols[lag] for lag in range(-predictive_length, 0)])
    future = encode_word([symbols[lag] for lag in range(1, predictive_length + 1)])
    alphabet = PARTITION_COUNT**predictive_length
    joint = past + alphabet * future
    return SymbolicItineraries(size, K, symbols, forward, backward, joint, past, future)
