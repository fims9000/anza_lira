import numpy as np

from lira_graph_cut_v2.graph_cut import connected, rasterize, tube_distance
from lira_h1.protocol import CUT_RADII, PROTOCOL
from lira_h1.ribbon import cumulative_arclength, flat_cap_ribbon, minimal_valid_ribbon_cut


def _straight_fixture():
    probability = np.zeros((72, 140), dtype=np.float32)
    probability[28:45, 8:132] = 0.9  # half-width eight around y=36
    trace = np.asarray([(36.0, float(x)) for x in range(8, 132)])
    start, end = 52, 76
    arc = cumulative_arclength(trace)
    left = trace[start - 8 : start]
    right = trace[end + 1 : end + 9]
    return probability, trace, start, end, arc, left, right


def test_h1_protocol_preserves_frozen_bank_and_locks() -> None:
    assert CUT_RADII == (3, 5, 7, 9, 11, 13, 15)
    assert PROTOCOL["dense"]["hard_threshold"] == 0.30
    assert PROTOCOL["dense"]["soft_thresholds"] == [0.12, 0.18, 0.24]
    assert PROTOCOL["locks"]["p0"] and PROTOCOL["locks"]["confirm_contents"]


def test_old_capsule_destroys_near_anchors_but_flat_ribbon_preserves_them() -> None:
    probability, trace, start, end, arc, left, right = _straight_fixture()
    hidden = trace[start : end + 1]
    old = tube_distance(hidden, probability.shape) <= 11.0
    new = flat_cap_ribbon(trace, arc[start], arc[end], 11.0, probability.shape)
    left_pixels = np.rint(left).astype(int)
    right_pixels = np.rint(right).astype(int)
    assert old[left_pixels[:, 0], left_pixels[:, 1]].any()
    assert old[right_pixels[:, 0], right_pixels[:, 1]].any()
    assert not new[left_pixels[:, 0], left_pixels[:, 1]].any()
    assert not new[right_pixels[:, 0], right_pixels[:, 1]].any()


def test_flat_ribbon_disconnects_band_without_longitudinal_spillover() -> None:
    probability, trace, start, end, arc, left, right = _straight_fixture()
    ribbon = flat_cap_ribbon(trace, arc[start], arc[end], 9.0, probability.shape)
    cut = (probability >= 0.12) & ~ribbon
    assert not connected(cut, rasterize(left, cut.shape, 1), rasterize(right, cut.shape, 1))
    assert not ribbon[:, : int(trace[start, 1])].any()
    assert not ribbon[:, int(trace[end, 1]) + 1 :].any()


def test_minimal_ribbon_cut_is_valid_and_preserves_context() -> None:
    probability, trace, start, end, arc, left, right = _straight_fixture()
    result, ribbon, cut = minimal_valid_ribbon_cut(
        probability, trace, arc[start], arc[end], left, right, trace[:start], trace[end + 1 :],
        np.zeros_like(probability, dtype=bool),
    )
    assert result.status == "VALID"
    assert result.radius == 9
    assert result.left_context_supported >= 8 and result.right_context_supported >= 8
    assert ribbon is not None and cut is not None


def test_curved_trace_respects_arclength_caps() -> None:
    angle = np.linspace(-0.9, 0.9, 101)
    trace = np.stack((48 + 25 * np.sin(angle), 64 + 25 * np.cos(angle)), axis=1)
    arc = cumulative_arclength(trace)
    start, end = 35, 65
    ribbon = flat_cap_ribbon(trace, arc[start], arc[end], 7.0, (100, 110))
    before = np.rint(trace[:start]).astype(int)
    after = np.rint(trace[end + 1 :]).astype(int)
    assert not ribbon[before[:, 0], before[:, 1]].any()
    assert not ribbon[after[:, 0], after[:, 1]].any()
    # Rounded endpoint pixels can project infinitesimally outside the cap; the
    # strict interior must be present and exterior trace pixels must remain.
    interior = np.rint(trace[start + 1 : end]).astype(int)
    assert ribbon[interior[:, 0], interior[:, 1]].all()


def test_reversal_invariance() -> None:
    probability, trace, start, end, arc, _left, _right = _straight_fixture()
    forward = flat_cap_ribbon(trace, arc[start], arc[end], 11.0, probability.shape)
    reversed_trace = trace[::-1].copy()
    reversed_arc = cumulative_arclength(reversed_trace)
    n = len(trace)
    reverse_start, reverse_end = n - 1 - end, n - 1 - start
    backward = flat_cap_ribbon(reversed_trace, reversed_arc[reverse_start], reversed_arc[reverse_end], 11.0, probability.shape)
    assert np.array_equal(forward, backward)


def test_parallel_collateral_is_rejected() -> None:
    probability, trace, start, end, arc, left, right = _straight_fixture()
    other = np.zeros_like(probability, dtype=bool)
    other[39, 52:85] = True
    result, _ribbon, _cut = minimal_valid_ribbon_cut(
        probability, trace, arc[start], arc[end], left, right, trace[:start], trace[end + 1 :], other,
    )
    assert result.status == "INVALID_COLLATERAL_TRACE"
    assert result.collateral_fraction > 0.05


def test_parent_v2_stop_remains_immutable() -> None:
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root / "results/lira_graph_cut_v2/benchmark/retention.json").read_text())
    assert parent["status"] == "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE"
    assert parent["splits"]["gc_development"]["valid_cases"] == 0
