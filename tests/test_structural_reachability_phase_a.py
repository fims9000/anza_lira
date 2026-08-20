import numpy as np
import pytest

from structural_reachability.phase_a import _line_pixels, score_candidate_path


def _fields() -> dict[str, np.ndarray]:
    shape = (1, 3, 4)
    return {
        "image": np.zeros((3, 3, 4), dtype=np.float32),
        "probability": np.full((3, 4), 0.81, dtype=np.float32),
        "membership": np.ones(shape, dtype=np.float32),
        "theta": np.zeros(shape, dtype=np.float32),
        "sigma_parallel": np.full(shape, 2.0, dtype=np.float32),
        "sigma_perpendicular": np.full(shape, 0.5, dtype=np.float32),
    }


def test_fixed_corridor_scores_are_finite_bounded_and_shared() -> None:
    path = _line_pixels((1, 0), (1, 3))
    result = score_candidate_path(_fields(), path)
    assert result["edge_count"] == 3
    assert result["A0_probability_only"] == pytest.approx(0.81)
    assert result["A1_rgb_similarity"] == pytest.approx(1.0)
    for key in ("A2_geometry_G_theta", "A3_geometry_plus_fuzzy", "A4_full_geometry"):
        assert 0.0 <= result[key] <= 1.0


def test_corridor_construction_is_deterministic_and_connected() -> None:
    first = _line_pixels((0, 0), (3, 2))
    second = _line_pixels((0, 0), (3, 2))
    assert first == second
    assert first[0] == (0, 0) and first[-1] == (3, 2)
    assert all(max(abs(y1 - y0), abs(x1 - x0)) == 1 for (y0, x0), (y1, x1) in zip(first, first[1:]))
