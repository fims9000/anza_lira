from __future__ import annotations

import math

import numpy as np

from trace_extraction.geometry import axial_distance
from trace_extraction.graph import extract_trace_graph, merge_trace_gaps, pair_junction_branches


def test_straight_line_is_one_trace_with_two_endpoints() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, 3:18] = True
    graph = extract_trace_graph(skeleton)
    assert len(graph.segments) == 1
    assert len(graph.endpoints) == 2
    assert len(graph.junctions) == 0


def test_two_disconnected_lines_are_two_trace_objects() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[5, 2:9] = True
    skeleton[15, 12:19] = True
    graph = extract_trace_graph(skeleton)
    assert len(graph.segments) == 2


def test_t_junction_is_one_cluster_with_three_branches() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    skeleton[10, 3:18] = True
    skeleton[3:11, 10] = True
    graph = extract_trace_graph(skeleton)
    assert len(graph.junctions) == 1
    assert len(graph.endpoints) == 3
    assert len(graph.segments) == 3


def test_x_crossing_has_four_branches_and_axial_pairing() -> None:
    skeleton = np.zeros((21, 21), dtype=bool)
    for offset in range(-7, 8):
        skeleton[10 + offset, 10 + offset] = True
        skeleton[10 + offset, 10 - offset] = True
    graph = extract_trace_graph(skeleton)
    assert len(graph.junctions) == 1
    assert len(graph.endpoints) == 4
    assert len(graph.segments) == 4
    pairs = pair_junction_branches(graph, junction_index=0, tangent_pixels=5)
    assert len(pairs) == 2
    assert all(math.degrees(pair.axial_error) < 5.0 for pair in pairs)


def test_border_endpoint_and_junction_flags_use_five_pixel_margin() -> None:
    skeleton = np.zeros((31, 31), dtype=bool)
    skeleton[3, 0:16] = True
    skeleton[3:17, 10] = True
    graph = extract_trace_graph(skeleton, border_margin=5)
    assert len(graph.junctions) == 1
    assert graph.junction_border_truncated == (True,)
    assert sum(graph.endpoint_border_truncated) >= 1
    assert any(segment.start_border_truncated or segment.end_border_truncated for segment in graph.segments)


def test_validation_gap_threshold_merges_small_but_not_large_collinear_gap() -> None:
    skeleton = np.zeros((31, 51), dtype=bool)
    skeleton[15, 4:21] = True
    skeleton[15, 25:45] = True
    segments = extract_trace_graph(skeleton).segments
    assert len(segments) == 2
    assert len(merge_trace_gaps(segments, max_gap_px=5.0)) == 1
    assert len(merge_trace_gaps(segments, max_gap_px=3.0)) == 2


def test_parallel_independent_lines_are_not_merged_by_gap_rule() -> None:
    skeleton = np.zeros((41, 41), dtype=bool)
    skeleton[12, 5:20] = True
    skeleton[26, 5:20] = True
    segments = extract_trace_graph(skeleton).segments
    assert len(segments) == 2
    assert len(merge_trace_gaps(segments, max_gap_px=20.0)) == 2
