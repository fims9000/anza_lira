from __future__ import annotations

import numpy as np
import pytest

from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v3 import sample_seed_v3
from synthetic.crossing_trace_bench_v4 import (
    LOCAL8_OFFSETS,
    RADIUS2_OFFSETS,
    benchmark_v4_config,
    generate_sample_v4,
    sample_seed_v4,
)


def test_v4_is_independent_deterministic_and_test_locked() -> None:
    first = generate_sample_v4("validation", 17, image_size=64)
    second = generate_sample_v4("validation", 17, image_size=64)
    assert sample_seed_v4("validation", 17) != sample_seed_v3("validation", 17)
    assert np.array_equal(first["image"], second["image"])
    assert benchmark_v4_config()["legacy_v3_selection_reused"] is False
    with pytest.raises(PermissionError, match="LOCKED_UNOPENED"):
        generate_sample_v4("test", 0)


def test_x_and_y_allow_multiple_outgoing_positive_edges() -> None:
    for index in (256, 258):
        sample = generate_sample_v4("validation", index, image_size=96)
        target = build_affinity_targets(sample, LOCAL8_OFFSETS)
        point = sample["junctions"][0]["point_xy"]
        x, y = int(round(point[0])), int(round(point[1]))
        assert int(target["affinity_positive"][:, y, x].sum()) >= 2


def test_negative_gap_has_hard_negative_edges_but_no_positive_lineage_edges() -> None:
    sample = generate_sample_v4("validation", 128, image_size=96)
    target = build_affinity_targets(sample, (*LOCAL8_OFFSETS, *RADIUS2_OFFSETS))
    corridor = sample["negative_gap_mask"]
    assert target["affinity_hard_negative"][:, corridor].any()
    assert not target["affinity_positive"][:, corridor].any()
