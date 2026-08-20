from __future__ import annotations

import numpy as np
import pytest

from method_repair.crowd_audit import (
    directed_positive_distances,
    stable_section_sample,
    tolerant_pair_statistics,
)


def test_shifted_thin_lines_have_low_pixel_dice_but_small_geometric_distance() -> None:
    first = np.zeros((24, 24), dtype=bool)
    second = np.zeros_like(first)
    first[3:21, 8] = True
    second[3:21, 11] = True
    stats = tolerant_pair_statistics(first, second)
    assert stats["pixel_dice"] == 0.0
    assert stats["distance_median_px"] == pytest.approx(3.0)
    assert stats["within_5px_fraction"] == 1.0
    assert stats["displaced_2_to_5px_fraction"] == 1.0


def test_missing_destination_is_explicitly_infinite() -> None:
    source = np.zeros((5, 5), dtype=bool)
    source[2, 2] = True
    distances = directed_positive_distances(source, np.zeros_like(source))
    assert distances.shape == (1,)
    assert np.isinf(distances[0])


def test_stable_section_sample_is_order_invariant_and_bounded() -> None:
    expected = stable_section_sample([5, 2, 9, 1], count=3)
    assert expected == stable_section_sample([1, 9, 2, 5], count=3)
    assert len(expected) == 3
    assert stable_section_sample([1, 2], count=40) == [1, 2]
