from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_mask_reconstruction import RepairProposal, apply_approved_paths


def pair_path(x0=3, x1=9):
    return RepairProposal("PAIR", (tuple((x, 5, 5) for x in range(x0, x1 + 1)),))


def test_pair_connects_two_only_and_preserves_original():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = True
    mask[9, 5, 5] = True
    original = mask.copy()
    result = apply_approved_paths(mask, np.eye(4), [pair_path()],
                                  radius_mm=0.0, max_new_voxels=20)
    assert result.accepted and result.reason == "ok"
    assert (result.original_components, result.final_components, result.added_voxels) == (2, 1, 5)
    assert result.mask[3:10, 5, 5].all()
    assert np.array_equal(mask, original)


def test_junction_three_arms_connect():
    mask = np.zeros((12, 12, 12), dtype=bool)
    starts = [(2, 6, 6), (6, 2, 6), (10, 6, 6)]
    for point in starts:
        mask[point] = True
    arms = [
        tuple((x, 6, 6) for x in range(2, 7)),
        tuple((6, y, 6) for y in range(2, 7)),
        tuple((x, 6, 6) for x in range(10, 5, -1)),
    ]
    result = apply_approved_paths(mask, np.eye(4),
                                  [RepairProposal("JUNCTION", tuple(arms))],
                                  radius_mm=0, max_new_voxels=30)
    assert result.accepted and (result.original_components, result.final_components) == (3, 1)


def test_unrelated_original_component_causes_atomic_abstain():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = mask[6, 6, 5] = True
    result = apply_approved_paths(mask, np.eye(4), [pair_path()],
                                  radius_mm=0, max_new_voxels=30)
    assert not result.accepted
    assert result.reason == "unselected_component_contact"
    assert np.array_equal(result.mask, mask)


def test_anisotropic_affine_uses_physical_mm_not_voxel_radius():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = True
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    result = apply_approved_paths(mask, affine, [pair_path()],
                                  radius_mm=1.1, max_new_voxels=100)
    assert result.accepted
    assert result.mask[6, 6, 5]
    assert not result.mask[6, 5, 6]


def test_ct_support_and_volume_budget_fail_closed():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = True
    no_support = np.ones_like(mask)
    no_support[6, 5, 5] = False
    denied = apply_approved_paths(mask, np.eye(4), [pair_path()], radius_mm=0,
                                  max_new_voxels=20, allowed_new_voxels=no_support)
    excess = apply_approved_paths(mask, np.eye(4), [pair_path()], radius_mm=0,
                                  max_new_voxels=3)
    assert not denied.accepted and denied.reason == "ct_support_rejected_new_voxels"
    assert not excess.accepted and excess.reason == "volume_budget_exceeded"
    assert np.array_equal(denied.mask, mask) and np.array_equal(excess.mask, mask)


def test_invalid_diagonal_jump_and_nonsensical_junction_rejected():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = True
    with pytest.raises(ValueError, match="non-26-connected"):
        apply_approved_paths(mask, np.eye(4),
            [RepairProposal("PAIR", (((3, 5, 5), (9, 5, 5)),))],
            radius_mm=0, max_new_voxels=20)
    with pytest.raises(ValueError, match="requires 3"):
        apply_approved_paths(mask, np.eye(4),
            [RepairProposal("JUNCTION", (((3, 5, 5), (4, 5, 5)),))],
            radius_mm=0, max_new_voxels=20)


def test_both_separate_families_allowed_but_contact_rejected():
    mask = np.zeros((22, 22, 22), dtype=bool)
    mask[3, 5, 5] = mask[9, 5, 5] = True
    starts = [(3, 16, 16), (7, 12, 16), (11, 16, 16)]
    for s in starts:
        mask[s] = True
    junction = RepairProposal("JUNCTION", (
        tuple((x, 16, 16) for x in range(3, 8)),
        tuple((7, y, 16) for y in range(12, 17)),
        tuple((x, 16, 16) for x in range(11, 6, -1)),
    ))
    result = apply_approved_paths(mask, np.eye(4), [pair_path(), junction], radius_mm=0,
                                  max_new_voxels=40)
    assert result.accepted and (result.original_components, result.final_components) == (5, 2)
    separate = mask.copy()
    separate[3, 16, 16] = separate[7, 12, 16] = separate[11, 16, 16] = False
    separate[3, 6, 5] = separate[7, 9, 5] = separate[11, 6, 5] = True
    nearby = RepairProposal("JUNCTION", (
        tuple((x, 6, 5) for x in range(3, 8)),
        tuple((7, y, 5) for y in range(9, 5, -1)),
        tuple((x, 6, 5) for x in range(11, 6, -1)),
    ))
    rejected = apply_approved_paths(separate, np.eye(4), [pair_path(), nearby],
                                    radius_mm=0, max_new_voxels=40)
    assert not rejected.accepted
