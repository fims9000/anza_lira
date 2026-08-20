from __future__ import annotations

import numpy as np
import pytest

from synthetic.crossing_trace_bench_v3 import (
    PAIRED_GAP_COUNT,
    SPLIT_SIZES_V3,
    benchmark_v3_config,
    generate_sample_v3,
)


def test_v3_is_deterministic_and_test_is_fail_closed() -> None:
    first = generate_sample_v3("validation", 17, image_size=64)
    second = generate_sample_v3("validation", 17, image_size=64)
    assert np.array_equal(first["image"], second["image"])
    assert np.array_equal(first["gate_target"], second["gate_target"])
    assert benchmark_v3_config()["legacy_test_stream"] == "IMMUTABLE_NOT_REUSED"
    assert benchmark_v3_config()["v2_stream"] == "IMMUTABLE_NOT_REUSED"
    with pytest.raises(PermissionError, match="LOCKED_UNOPENED"):
        generate_sample_v3("test", 0)


def test_validation_schedule_has_required_paired_gap_counts() -> None:
    assert SPLIT_SIZES_V3["validation"] == 512
    cases = [generate_sample_v3("validation", index, image_size=48)["case"] for index in range(256)]
    assert cases.count("fault_with_gap") == PAIRED_GAP_COUNT == 128
    assert cases.count("negative_gap") == PAIRED_GAP_COUNT == 128


def test_positive_and_negative_gap_pairs_are_geometrically_matched() -> None:
    positive = generate_sample_v3("validation", 11, image_size=96)
    negative = generate_sample_v3("validation", 128 + 11, image_size=96)
    assert positive["pair_id"] == negative["pair_id"] == 11
    for key in (
        "gap_length_px",
        "endpoint_distance_px",
        "local_axial_orientation_rad",
        "geometry_seed",
        "render_difficulty_seed",
    ):
        assert positive["gap_match"][key] == pytest.approx(negative["gap_match"][key])
    assert positive["positive_gap_mask"].any()
    assert negative["negative_gap_mask"].any()
    assert positive["gaps"][0]["latent_instance_id"] == 1
    assert negative["gaps"][0]["latent_instance_id"] is None


def test_context_stratum_contains_required_topologies_and_hard_cases() -> None:
    cases = {
        generate_sample_v3("validation", index, image_size=48)["case"]
        for index in range(256, 512)
    }
    assert {
        "x_junction",
        "t_junction",
        "y_junction",
        "acute_angle_crossing",
        "near_parallel",
        "curved_fault",
        "curved_crossing",
        "similar_tangent_crossing",
        "nontrivial_pairing",
    } <= cases


def test_gate_target_is_contextual_and_hard_negatives_remain_zero() -> None:
    junction = generate_sample_v3("validation", 256, image_size=96)
    point = junction["junctions"][0]["point_xy"]
    x, y = int(round(point[0])), int(round(point[1]))
    assert junction["gate_target"][y, x] == pytest.approx(1.0)
    assert np.count_nonzero(junction["gate_target"] > 0.1) > np.count_nonzero(junction["junction_map"])

    near_parallel_index = 256 + 4
    hard_negative = generate_sample_v3("validation", near_parallel_index, image_size=96)
    assert hard_negative["case"] == "near_parallel"
    assert not hard_negative["gate_target"].any()
    assert hard_negative["gate_hard_negative_mask"].any()
