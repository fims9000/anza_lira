import numpy as np
import json
from pathlib import Path

from lira_graph_cut_v2.benchmark import split_manifest
from lira_graph_cut_v2.graph_cut import connected, minimal_valid_cut, rasterize
from lira_graph_cut_v2.protocol import CUT_RADII, PROTOCOL


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.zeros((64, 128), dtype=np.float32)
    probability[28:37, 8:120] = 0.9
    hidden = np.asarray([(32.0, float(x)) for x in range(60, 69)])
    left_anchor = np.asarray([(32.0, float(x)) for x in range(52, 60)])
    right_anchor = np.asarray([(32.0, float(x)) for x in range(69, 77)])
    left_context = np.asarray([(32.0, float(x)) for x in range(8, 60)])
    right_context = np.asarray([(32.0, float(x)) for x in range(69, 120)])
    return probability, hidden, left_anchor, right_anchor, left_context, right_context


def test_graph_cut_split_is_fresh_disjoint_and_confirm_locked() -> None:
    manifest = split_manifest()
    sets = [set(value) for value in manifest["splits"].values()]
    assert all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert manifest["confirm_contents_opened"] is False
    assert PROTOCOL["locks"]["p0"] and PROTOCOL["locks"]["expert"]
    assert tuple(PROTOCOL["treatment"]["candidate_radii_px"]) == CUT_RADII


def test_minimal_cut_selects_first_disconnecting_radius() -> None:
    probability, hidden, left, right, left_context, right_context = _fixture()
    result, tube, support = minimal_valid_cut(
        probability, hidden, left, right, left_context, right_context, np.zeros_like(probability, dtype=bool)
    )
    assert result.status == "VALID"
    assert result.radius == 5
    assert tube is not None and support is not None
    assert connected(probability >= 0.12, rasterize(left, probability.shape, 1), rasterize(right, probability.shape, 1))
    assert not connected(support, rasterize(left, probability.shape, 1), rasterize(right, probability.shape, 1))


def test_pre_disconnected_case_is_not_a_treatment() -> None:
    probability, hidden, left, right, left_context, right_context = _fixture()
    probability[:, 63:66] = 0.0
    result, tube, support = minimal_valid_cut(
        probability, hidden, left, right, left_context, right_context, np.zeros_like(probability, dtype=bool)
    )
    assert result.status == "INELIGIBLE_PRE_DISCONNECTED"
    assert tube is None and support is None


def test_collateral_trace_exclusion_precedes_sbpp() -> None:
    probability, hidden, left, right, left_context, right_context = _fixture()
    collateral = np.zeros_like(probability, dtype=bool)
    collateral[25:40, 56:73] = True
    result, _tube, _support = minimal_valid_cut(
        probability, hidden, left, right, left_context, right_context, collateral
    )
    assert result.status == "INVALID_COLLATERAL_TRACE"
    assert result.collateral_fraction > 0.05


def test_frozen_benchmark_stop_keeps_candidate_and_p0_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/lira_graph_cut_v2/benchmark/retention.json").read_text())
    assert result["status"] == "STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE"
    assert result["splits"]["gc_development"]["retention"] < 0.50
    assert result["splits"]["gc_development"]["valid_cases"] == 0
    assert result["splits"]["gc_development"]["treatment_validity"] is None
    candidate = (root / "results/lira_graph_cut_v2/candidate/LIRA_GRAPH_CUT_CANDIDATE_REPORT.md").read_text()
    assert "LOCKED_NOT_RUN_AFTER_STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE" in candidate
    assert not (root / "results/lira_graph_cut_v2/relation/checkpoint.pt").exists()
