from __future__ import annotations

import numpy as np
import pytest
import torch

from synthetic.crossing_trace_bench import CrossingTraceBench, generate_sample, sample_seed


def test_split_rng_streams_are_disjoint_and_frozen() -> None:
    assert sample_seed("train", 0) == 10_000_000
    assert sample_seed("validation", 0) == 20_000_000
    assert sample_seed("test", 0) == 30_000_000
    assert len({sample_seed(split, 17) for split in ("train", "validation", "test")}) == 3


def test_generation_is_deterministic_and_seismic_like() -> None:
    first = generate_sample("train", 3, case="curved_fault")
    second = generate_sample("train", 3, case="curved_fault")
    assert np.array_equal(first["image"], second["image"])
    assert first["image"].shape == (3, 128, 128)
    assert np.isfinite(first["image"]).all()
    assert 0.0 <= first["image"].min() <= first["image"].max() <= 1.0
    assert np.unique(first["image"][0]).size > 100
    assert float(first["image"].std()) > 0.05


@pytest.mark.parametrize(
    ("case", "branches", "pairings"),
    [
        ("x_junction", 4, 2),
        ("t_junction", 3, 1),
        ("y_junction", 3, 2),
    ],
)
def test_junction_cases_have_explicit_topology_relation(case: str, branches: int, pairings: int) -> None:
    sample = generate_sample("validation", 1, case=case)
    assert sample["branch_masks"].shape[0] == branches
    assert sample["junction_map"].any()
    assert len(sample["junctions"]) == 1
    assert len(sample["junctions"][0]["incident_branch_ids"]) == branches
    assert len(sample["junctions"][0]["continuation_relation"]) == pairings
    assert int(sample["continuation_relation_matrix"].sum()) == 2 * pairings


def test_x_crossing_preserves_both_instance_identities_at_overlap() -> None:
    sample = generate_sample("validation", 2, case="x_junction")
    assert sample["instance_masks"].shape[0] == 2
    assert sample["instance_overlap_mask"].any()
    overlap = sample["instance_overlap_mask"]
    assert np.all(sample["instance_masks"][:, overlap])
    assert sample["fault_instance_ids"] == [1, 2]


def test_parallel_and_close_nonintersecting_lines_are_not_joined() -> None:
    for case in ("near_parallel", "close_non_intersecting"):
        sample = generate_sample("validation", 4, case=case)
        assert not sample["junction_map"].any()
        assert not sample["continuation_relation_matrix"].any()
        assert sample["instance_masks"].shape[0] == 2


def test_gap_has_one_identity_two_visible_branches_and_exact_gap_mask() -> None:
    sample = generate_sample("validation", 5, case="fault_with_gap")
    assert sample["instance_masks"].shape[0] == 1
    assert sample["branch_masks"].shape[0] == 2
    assert sample["gap_mask"].any()
    assert not np.any(sample["visible_fault_mask"] & sample["gap_mask"])
    assert np.all(sample["latent_fault_mask"][sample["gap_mask"]])
    assert np.array_equal(
        sample["gap_mask"], sample["latent_fault_mask"] & ~sample["visible_fault_mask"]
    )


def test_dataset_tensor_contract_and_finite_ranges() -> None:
    sample = CrossingTraceBench("train", length=2)[0]
    assert sample["image"].shape == (3, 128, 128)
    assert sample["visible_fault_mask"].shape == (1, 128, 128)
    assert sample["latent_fault_mask"].shape == (1, 128, 128)
    assert sample["continuation_relation_matrix"].ndim == 2
    assert torch.isfinite(sample["image"]).all()
    assert torch.isfinite(sample["branch_tangent_cos2"]).all()
    assert torch.isfinite(sample["branch_tangent_sin2"]).all()
