from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from synthetic.crossing_trace_bench import generate_sample
from synthetic.geometry_generator import (
    NONTRIVIAL_PAIRING_CASES,
    generate_geometry,
    scale_geometry,
)
from synthetic.instance_targets import rasterize_targets


def _axial_angle_between_outward_rays(sample: dict, first_id: int, second_id: int) -> float:
    geometry = scale_geometry(
        generate_geometry(sample["case"], np.random.default_rng(sample["seed"])),
        sample["image_size"],
    )
    junction_xy = np.asarray(geometry.junctions[0].point_xy)

    def outward(branch_id: int) -> np.ndarray:
        branch = next(value for value in geometry.branches if value.branch_id == branch_id)
        points = branch.points_xy
        if np.linalg.norm(points[0] - junction_xy) <= np.linalg.norm(points[-1] - junction_xy):
            vector = points[1] - points[0]
        else:
            vector = points[-2] - points[-1]
        return vector / np.linalg.norm(vector)

    dot = abs(float(np.dot(outward(first_id), outward(second_id))))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


@pytest.mark.parametrize("case", ("x_junction", "curved_crossing", "nontrivial_pairing"))
def test_crossing_pixel_can_belong_to_two_latent_instance_masks(case: str) -> None:
    sample = generate_sample("validation", 101, case=case)
    overlap = np.asarray(sample["instance_overlap_mask"], dtype=bool)
    assert overlap.any()
    assert np.all(np.asarray(sample["instance_masks"])[:, overlap].sum(axis=0) >= 2)


@pytest.mark.parametrize(
    "case",
    ("single_straight", "fault_with_gap", "negative_gap", "x_junction", "y_junction"),
)
def test_visible_latent_and_gap_semantics_are_exact(case: str) -> None:
    sample = generate_sample("validation", 102, case=case)
    visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
    latent = np.asarray(sample["latent_fault_mask"], dtype=bool)
    gap = np.asarray(sample["gap_mask"], dtype=bool)
    assert np.all(~visible | latent)
    assert np.array_equal(gap, latent & ~visible)


def test_positive_gap_belongs_to_exactly_one_known_latent_instance() -> None:
    sample = generate_sample("validation", 103, case="fault_with_gap")
    gap = np.asarray(sample["positive_gap_mask"], dtype=bool)
    owners = np.asarray(sample["positive_gap_owner"])[gap]
    membership = np.asarray(sample["instance_masks"], dtype=bool)[:, gap].sum(axis=0)
    assert gap.any()
    assert set(np.unique(owners)) == {1}
    assert np.all(membership == 1)
    assert sample["gaps"][0]["latent_instance_id"] == 1


def test_negative_gap_has_no_common_latent_instance() -> None:
    sample = generate_sample("validation", 104, case="negative_gap")
    gap = np.asarray(sample["negative_gap_mask"], dtype=bool)
    assert gap.any()
    assert not np.asarray(sample["instance_masks"], dtype=bool)[:, gap].any()
    assert sample["branch_instance_id"] == [1, 2]
    assert sample["gaps"][0]["latent_instance_id"] is None


def test_x_lineage_relation_survives_arbitrary_branch_ordering() -> None:
    rng = np.random.default_rng(105)
    geometry = scale_geometry(generate_geometry("x_junction", rng), 128)
    original = rasterize_targets(geometry, 128)
    reordered = rasterize_targets(replace(geometry, branches=tuple(reversed(geometry.branches))), 128)
    assert original["branch_ids"] == reordered["branch_ids"]
    assert np.array_equal(
        original["continuation_relation_matrix"], reordered["continuation_relation_matrix"]
    )
    assert original["junctions"][0]["continuation_relation"] == [[1, 2], [3, 4]]


def test_nontrivial_pairing_truth_intentionally_disagrees_with_minimum_angle() -> None:
    sample = generate_sample("validation", 106, case="nontrivial_pairing")
    relation = np.asarray(sample["continuation_relation_matrix"], dtype=bool)
    branch_index = {branch_id: index for index, branch_id in enumerate(sample["branch_ids"])}
    assert relation[branch_index[1], branch_index[2]]
    assert not relation[branch_index[1], branch_index[3]]
    assert _axial_angle_between_outward_rays(sample, 1, 3) < _axial_angle_between_outward_rays(sample, 1, 2)


def test_curved_crossing_relation_is_generator_lineage() -> None:
    sample = generate_sample("validation", 107, case="curved_crossing")
    assert sample["branch_instance_id"] == [1, 1, 2, 2]
    assert sample["junctions"][0]["continuation_relation"] == [[1, 2], [3, 4]]


def test_t_and_y_have_different_topology_contracts() -> None:
    t_sample = generate_sample("validation", 108, case="t_junction")
    y_sample = generate_sample("validation", 108, case="y_junction")
    t_junction = t_sample["junctions"][0]
    y_junction = y_sample["junctions"][0]
    assert t_junction["junction_type"] == "t_intersection"
    assert t_junction["incident_instance_ids"] == [1, 1, 2]
    assert t_junction["continuation_relation"] == [[1, 2]]
    assert y_junction["junction_type"] == "y_branch"
    assert y_junction["incident_instance_ids"] == [1, 1, 1]
    assert y_junction["continuation_relation"] == [[1, 2], [1, 3]]


@pytest.mark.parametrize("case", NONTRIVIAL_PAIRING_CASES)
def test_ambiguous_crossings_are_in_the_nontrivial_pairing_stratum(case: str) -> None:
    sample = generate_sample("validation", 109, case=case)
    assert "nontrivial_pairing" in sample["strata"]


def test_nearby_junction_uses_a_valid_three_branch_t_contract() -> None:
    sample = generate_sample("validation", 110, case="crossing_near_junction")
    assert [junction["junction_type"] for junction in sample["junctions"]] == [
        "x_crossing",
        "t_intersection",
    ]
    assert len(sample["junctions"][1]["incident_branch_ids"]) == 3
