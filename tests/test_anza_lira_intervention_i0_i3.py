from pathlib import Path

import numpy as np

from lira_final.data.cracks_trace_audit import CrowdTrace
from lira_intervention.candidate import masked_probability
from lira_intervention.data import build_interventions, make_intervention, split_manifest
from lira_intervention.protocol import PROTOCOL


def test_intervention_splits_are_disjoint_and_confirm_locked() -> None:
    manifest = split_manifest()
    sets = [set(value) for value in manifest["splits"].values()]
    assert all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert manifest["confirm_contents_opened"] is False
    assert PROTOCOL["locks"]["expert"] and PROTOCOL["locks"]["confirm_contents"]


def test_intervention_is_deterministic_internal_and_one_per_trace() -> None:
    points = np.asarray([(30.0, float(x)) for x in range(100)], dtype=np.float32)
    trace = CrowdTrace(1, "novice12", "trace-a", points)
    first = make_intervention(trace, "ig_development", 24)
    second = make_intervention(trace, "ig_development", 24)
    assert first == second and first is not None
    assert first.gap_length_px == 24
    assert len(build_interventions("ig_development", (trace,))) == 1
    assert first.source_yx[1] >= 12 and first.destination_context_yx[-1][1] <= 99


def test_only_dense_evidence_tube_is_erased() -> None:
    points = np.asarray([(20.0, float(x)) for x in range(100)], dtype=np.float32)
    case = make_intervention(CrowdTrace(1, "novice12", "trace-b", points), "ig_development", 16)
    assert case is not None
    probability = np.ones((64, 128), dtype=np.float32)
    masked, tube = masked_probability(probability, case)
    assert np.all(masked[tube] == 0.0)
    assert np.all(masked[~tube] == probability[~tube])
    image = np.random.default_rng(1).random((64, 128, 3), dtype=np.float32)
    before = image.copy()
    masked_probability(probability, case)
    assert np.array_equal(image, before)


def test_frozen_i2_stop_keeps_i3_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = __import__("json").loads((root / "results/lira_intervention_final/i2_candidate/summary.json").read_text())
    assert summary["candidate_recall_at_12"] < 0.87
    assert summary["status"] == "STOP_LIRA_INTERVENTION_CANDIDATE"
    locked = (root / "results/lira_intervention_final/i3_relation_s41/LIRA_INTERVENTION_RELATION_S41_REPORT.md").read_text()
    assert "LOCKED_NOT_RUN_AFTER_I2_STOP" in locked
    assert not (root / "results/lira_intervention_final/i3_relation_s41/checkpoint.pt").exists()
