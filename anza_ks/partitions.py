"""Fixed finite symbolic partitions; V1 deliberately is not a Markov partition."""

from __future__ import annotations

import numpy as np


PARTITION_COUNT = 4


def quadrant_partition(size: int = 17) -> np.ndarray:
    if size < 3:
        raise ValueError("partition size must be >=3")
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    split = size // 2
    return (x >= split).astype(np.int64) + 2 * (y >= split).astype(np.int64)
