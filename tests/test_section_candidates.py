"""Whole-section candidate graph tests using synthetic masks, not real labels."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from trace_extraction.section_candidates import (
    full_section_candidate_graph, multi_threshold_section_candidates,
)


def _two_fragments():
    p = np.zeros((96, 128), np.float32)
    p[48, 12:44] = .95
    p[48, 53:88] = .94
    return p


def test_full_section_finds_gap_without_expert_rectangle():
    p = _two_fragments()
    result = full_section_candidate_graph(p, threshold=.65, max_gap_px=20.)
    assert len(result.ports) == 4
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    endpoints = {result.ports[pair.first_port].pixel,
                 result.ports[pair.second_port].pixel}
    assert endpoints == {(48, 43), (48, 53)}
    assert pair.first_trace != pair.second_trace
    assert pair.source_alignment > .9 and pair.target_alignment > .9
    assert "rect" not in inspect.signature(full_section_candidate_graph).parameters
    assert "expert" not in inspect.signature(full_section_candidate_graph).parameters


def test_multi_threshold_keeps_faint_lines_as_separate_layers():
    p = _two_fragments()
    p[75, 20:51] = .42
    runs = multi_threshold_section_candidates(p)
    assert [r.threshold for r in runs] == [.65, .50, .35]
    assert len(runs[0].ports) == len(runs[1].ports)
    assert len(runs[2].ports) > len(runs[1].ports)
    assert np.array_equal(p, _two_fragments() + (p - _two_fragments()))


def test_perpendicular_nearby_track_not_forced_into_wrong_pair():
    p = _two_fragments()
    p[15:43, 49] = .96
    result = full_section_candidate_graph(
        p, threshold=.65, max_gap_px=20., min_alignment=.7)
    assert result.pairs
    assert all(pair.source_alignment >= .7 and pair.target_alignment >= .7
               for pair in result.pairs)
    assert any({result.ports[pair.first_port].pixel,
                result.ports[pair.second_port].pixel} ==
               {(48, 43), (48, 53)} for pair in result.pairs)


def test_border_and_dense_mask_fail_safe():
    p = _two_fragments()
    p[0, 5:34] = .97
    r = full_section_candidate_graph(p, threshold=.65, border_margin=5)
    assert all(port.pixel[0] >= 5 for port in r.ports)
    with pytest.raises(ValueError, match="Dense"):
        full_section_candidate_graph(np.ones((96,128), np.float32))
    with pytest.raises(ValueError, match="policy"):
        full_section_candidate_graph(p, max_neighbors_per_port=0)
    with pytest.raises(ValueError, match="distinct"):
        multi_threshold_section_candidates(p, (.65, .65))


def test_unmodified_probability_and_deterministic_candidates():
    p = _two_fragments()
    before = p.copy()
    a = full_section_candidate_graph(p)
    b = full_section_candidate_graph(p)
    assert a.ports == b.ports and a.pairs == b.pairs
    assert np.array_equal(p, before)
