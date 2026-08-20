from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from anza_hs.metrics import evaluate, sample_metrics
from anza_hs.stress_bench import generate_stress_sample


ROOT = Path(__file__).resolve().parents[1]
H0 = ROOT / "results" / "anza_hs" / "h0"
H1 = ROOT / "results" / "anza_hs" / "h1"


def test_perfect_visible_prediction_has_exact_overlap_and_topology():
    sample = generate_stress_sample("dev", 1)
    probability = np.asarray(sample["visible_fault_mask"], dtype=np.float32)
    metrics = sample_metrics(probability, sample, 0.5)
    assert metrics["dice"] == metrics["precision"] == metrics["recall"] == 1.0
    assert metrics["cldice"] == metrics["skeleton_f1"] == 1.0
    assert metrics["fragmentation"] == 0.0


def test_metric_aggregation_uses_sample_rows():
    samples = [generate_stress_sample("dev", index) for index in (44, 45)]
    probabilities = [np.asarray(sample["visible_fault_mask"], dtype=np.float32) for sample in samples]
    summary, rows = evaluate(probabilities, samples, 0.5)
    assert summary["sample_count"] == 2 and len(rows) == 2 and summary["overall"]["dice"] == 1.0


def test_h0_and_h1_provenance_are_frozen():
    h0 = json.loads((H0 / "validator.json").read_text()); metrics = json.loads((H1 / "metrics.json").read_text())
    assert h0["research_status"] == "ANZA_HS_H0_PASS"
    assert metrics["protocol_sha256"] == h0["protocol_sha256"]
    assert metrics["stressbench_sha256"] == h0["stressbench_sha256"]


def test_h1_matrix_is_complete_and_identical_budget():
    metrics = json.loads((H1 / "metrics.json").read_text())
    assert set(metrics["variants"]) == {"B0_backbone", "B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic"}
    assert {value["run"]["epoch"] for value in metrics["variants"].values()} == {20}
    assert {value["run"]["protocol_sha256"] for value in metrics["variants"].values()} == {metrics["protocol_sha256"]}


def test_h1_gate_status_is_computed_from_frozen_checks():
    metrics = json.loads((H1 / "metrics.json").read_text()); checks = metrics["comparison"]["gate_checks"]
    expected = checks["dice_noninferiority"] and (checks["cldice_gain"] or checks["fragmentation_reduction"])
    assert metrics["gate_pass"] is expected
    assert metrics["status"] == ("ANZA_HS_H1_PASS" if expected else "HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL")


def test_calibration_and_gate_indices_do_not_overlap():
    with (H1 / "raw_per_sample.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 880 and {int(row["index"]) for row in rows} == set(range(44, 264))
    freeze = json.loads((H1 / "threshold_freeze.json").read_text())
    assert freeze["grid"] == pytest.approx([0.20 + 0.05 * index for index in range(13)])


def test_b3_precision_was_matched_without_gate_feedback():
    freeze = json.loads((H1 / "threshold_freeze.json").read_text())
    b2 = freeze["selected"]["B2_generic_aniso"]["precision"]
    b3 = freeze["selected"]["B3_anza_hyperbolic"]["precision"]
    assert freeze["B3_match_target_precision"] == b2 and abs(b3 - b2) < 0.001


def test_no_downstream_or_tuning_access_after_h1():
    metrics = json.loads((H1 / "metrics.json").read_text())
    for key in ("confirm_opened", "test_opened", "H2_opened", "cracks_accessed", "continuation_trained", "expert_accessed", "lambda_tuned", "M_tuned", "base_scale_alternative_used"):
        assert metrics[key] is False
