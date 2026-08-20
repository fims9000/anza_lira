from __future__ import annotations

import numpy as np

from synthetic.crossing_trace_bench import sample_seed
from synthetic.crossing_trace_bench_v2 import (
    benchmark_v2_config,
    generate_sample_v2,
    sample_seed_v2,
)


def _junction_mode_count(case: str) -> int:
    sample = generate_sample_v2("validation", 0, image_size=96, case=case)
    point = sample["junctions"][0]["point_xy"]
    x, y = int(round(point[0])), int(round(point[1]))
    return int(sample["gt_mode_count"][y, x])


def test_v2_stream_is_deterministic_and_independent_from_frozen_old_stream() -> None:
    first = generate_sample_v2("validation", 17, image_size=64)
    second = generate_sample_v2("validation", 17, image_size=64)
    assert sample_seed_v2("validation", 17) != sample_seed("validation", 17)
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first["gt_theta_set"], second["gt_theta_set"])
    assert benchmark_v2_config()["old_test_stream"] == "IMMUTABLE_AND_NOT_REUSED"


def test_x_t_y_have_semantically_distinct_tangent_set_cardinality() -> None:
    assert _junction_mode_count("x_junction") == 2
    assert _junction_mode_count("t_junction") == 2
    assert _junction_mode_count("y_junction") == 3


def test_mode_count_matches_validity_and_positive_gap_has_latent_tangent() -> None:
    sample = generate_sample_v2("validation", 5, image_size=96, case="fault_with_gap")
    assert np.array_equal(sample["gt_mode_count"], sample["gt_theta_valid"].sum(axis=0))
    gap = sample["positive_gap_mask"]
    assert gap.any()
    assert np.all(sample["gt_mode_count"][gap] == 1)


def test_crossing_branch_tangents_retain_separate_lineage_fields() -> None:
    sample = generate_sample_v2("validation", 9, image_size=96, case="x_junction")
    assert sample["gt_branch_theta"].shape[0] == 4
    assert sample["gt_branch_theta_valid"].shape == sample["branch_masks"].shape
    assert sample["instance_overlap_mask"].any()
