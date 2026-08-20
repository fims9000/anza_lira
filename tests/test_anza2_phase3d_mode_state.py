import numpy as np
import torch

from anza2.phase3d.mode_state_graph import mode_state_edge_weights, permute_modes
from anza2.phase3d.mode_state_widest_path import exhaustive_mode_state_widest_path, mode_state_widest_path
from models.anza2.affinity import ANZA2StructuralAffinity
from models.anza2.field import ANZA2FieldOutput
from structural.widest_path import domain_restricted_widest_path


OFFSETS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _field(membership, angles, *, scale=0.75, hyper=1.5):
    membership = torch.as_tensor(membership, dtype=torch.float64).unsqueeze(0)
    angles = torch.as_tensor(angles, dtype=torch.float64).unsqueeze(0)
    orientation = torch.stack((torch.cos(2 * angles), torch.sin(2 * angles)), dim=2)
    base = torch.full_like(membership, scale); h = torch.full_like(membership, hyper)
    return ANZA2FieldOutput(membership, orientation, base, h, base * torch.exp(h), base * torch.exp(-h))


def test_single_mode_scalar_and_mode_state_are_equal():
    membership = np.full((1, 1, 5), 0.98); angles = np.zeros_like(membership)
    field = _field(membership, angles)
    scalar = ANZA2StructuralAffinity(OFFSETS4)(field)[0].numpy()
    state = mode_state_edge_weights(field, OFFSETS4)[0].numpy()
    domain = np.ones((1, 5), dtype=bool)
    scalar_result = domain_restricted_widest_path(scalar, (0, 0), (0, 4), domain=domain, offsets=OFFSETS4)
    state_result = mode_state_widest_path(state, [(0, 0, 0)], [(0, 4, 0)], domain=domain, offsets=OFFSETS4)
    assert np.isclose(scalar_result[0], state_result[0])


def test_x_mode_state_blocks_free_horizontal_to_vertical_turn():
    membership = np.full((2, 5, 5), 0.001); angles = np.zeros_like(membership)
    membership[0, 2, :] = 0.98; angles[0, 2, :] = 0.0
    membership[1, :, 2] = 0.98; angles[1, :, 2] = np.pi / 2
    field = _field(membership, angles)
    state = mode_state_edge_weights(field, OFFSETS4)[0].numpy()
    domain = (membership.max(axis=0) > 0.5)
    straight = mode_state_widest_path(state, [(2, 0, 0)], [(2, 4, 0)], domain=domain, offsets=OFFSETS4)
    wrong = mode_state_widest_path(state, [(2, 0, 0)], [(0, 2, 1)], domain=domain, offsets=OFFSETS4)
    assert straight[0] > 0.8
    assert wrong[0] < straight[0] * 0.25
    assert all(first[:2] != second[:2] for first, second in zip(wrong[1], wrong[1][1:]))


def test_vertical_entry_cannot_freely_turn_horizontal():
    membership = np.full((2, 5, 5), 0.001); angles = np.zeros_like(membership)
    membership[0, 2, :] = 0.98; membership[1, :, 2] = 0.98
    angles[1] = np.pi / 2
    field = _field(membership, angles)
    state = mode_state_edge_weights(field, OFFSETS4)[0].numpy(); domain = membership.max(axis=0) > 0.5
    straight = mode_state_widest_path(state, [(0, 2, 1)], [(4, 2, 1)], domain=domain, offsets=OFFSETS4)
    wrong = mode_state_widest_path(state, [(0, 2, 1)], [(2, 4, 0)], domain=domain, offsets=OFFSETS4)
    assert wrong[0] < straight[0] * 0.25


def test_curved_mode_sequence_remains_reachable():
    membership = np.full((1, 4, 4), 0.001); angles = np.zeros_like(membership)
    path = [(3, 0), (3, 1), (2, 1), (1, 1), (1, 2), (0, 2)]
    for (y, x), (ny, nx) in zip(path, path[1:]):
        membership[0, y, x] = 0.98; angles[0, y, x] = np.arctan2(ny - y, nx - x)
    membership[0, path[-1][0], path[-1][1]] = 0.98; angles[0, path[-1][0], path[-1][1]] = angles[0, path[-2][0], path[-2][1]]
    field = _field(membership, angles, hyper=0.5)
    state = mode_state_edge_weights(field, OFFSETS4)[0].numpy(); domain = membership[0] > 0.5
    result = mode_state_widest_path(state, [(3, 0, 0)], [(0, 2, 0)], domain=domain, offsets=OFFSETS4)
    assert result[0] > 0.05 and len(result[1]) == len(path)


def test_mode_permutation_preserves_best_path_score_and_weights_stay_bounded():
    membership = np.stack((np.full((2, 3), 0.8), np.full((2, 3), 0.4)))
    angles = np.stack((np.zeros((2, 3)), np.full((2, 3), np.pi / 2)))
    field = _field(membership, angles)
    first = mode_state_edge_weights(field, OFFSETS4)
    order = torch.tensor([1, 0]); second = mode_state_edge_weights(permute_modes(field, order), OFFSETS4)
    assert first.min() >= 0 and first.max() <= 1
    assert torch.allclose(first[:, :, order][:, :, :, order], second)


def test_optimized_widest_path_matches_bruteforce_and_ties_are_deterministic():
    edges = np.zeros((4, 1, 1, 2, 2), dtype=np.float32)
    # Every valid grid edge has the same score; lexicographic path is stable.
    for channel, (dx, dy) in enumerate(OFFSETS4):
        for y in range(2):
            for x in range(2):
                if 0 <= y + dy < 2 and 0 <= x + dx < 2:
                    edges[channel, 0, 0, y, x] = 0.7
    domain = np.ones((2, 2), dtype=bool)
    fast = mode_state_widest_path(edges, [(0, 0, 0)], [(1, 1, 0)], domain=domain, offsets=OFFSETS4)
    exact = exhaustive_mode_state_widest_path(edges, [(0, 0, 0)], [(1, 1, 0)], domain=domain, offsets=OFFSETS4)
    assert fast == exact
    assert fast == mode_state_widest_path(edges, [(0, 0, 0)], [(1, 1, 0)], domain=domain, offsets=OFFSETS4)
