import numpy as np

from structural.widest_path import domain_restricted_widest_path, exhaustive_widest_path


OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _set_edge(relation, first, second, weight):
    dy, dx = second[0] - first[0], second[1] - first[1]
    channel = OFFSETS.index((dx, dy))
    relation[channel, first[0], first[1]] = weight


def _undirected(relation, first, second, weight):
    _set_edge(relation, first, second, weight)
    _set_edge(relation, second, first, weight)


def test_t16_optimized_widest_path_matches_exhaustive_reference() -> None:
    relation = np.zeros((4, 2, 3), dtype=np.float32)
    _undirected(relation, (0, 0), (0, 1), 0.8)
    _undirected(relation, (0, 1), (0, 2), 0.7)
    _undirected(relation, (0, 0), (1, 0), 0.9)
    _undirected(relation, (1, 0), (1, 1), 0.9)
    _undirected(relation, (1, 1), (1, 2), 0.6)
    _undirected(relation, (1, 2), (0, 2), 0.9)
    domain = np.ones((2, 3), dtype=bool)
    fast = domain_restricted_widest_path(relation, (0, 0), (0, 2), domain=domain, offsets=OFFSETS)
    exact = exhaustive_widest_path(relation, (0, 0), (0, 2), domain=domain, offsets=OFFSETS)
    assert fast == exact
    assert fast[0] == np.float32(0.7)


def test_t17_domain_mask_blocks_stronger_background_shortcut() -> None:
    relation = np.zeros((4, 3, 5), dtype=np.float32)
    for x in range(4):
        _undirected(relation, (1, x), (1, x + 1), 0.55)
    _undirected(relation, (1, 0), (0, 0), 0.95)
    for x in range(4):
        _undirected(relation, (0, x), (0, x + 1), 0.95)
    _undirected(relation, (0, 4), (1, 4), 0.95)
    full = np.ones((3, 5), dtype=bool)
    corridor = np.zeros((3, 5), dtype=bool)
    corridor[1] = True
    unrestricted = domain_restricted_widest_path(relation, (1, 0), (1, 4), domain=full, offsets=OFFSETS)
    restricted = domain_restricted_widest_path(relation, (1, 0), (1, 4), domain=corridor, offsets=OFFSETS)
    assert unrestricted[0] > restricted[0]
    assert restricted[0] == np.float32(0.55)
    assert all(y == 1 for y, _x in restricted[1])
