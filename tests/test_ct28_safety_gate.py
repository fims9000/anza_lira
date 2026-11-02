"""Regression checks for CT28 safety policy and saved validation pack.

No medical volumes or held-out test subjects are accessed.
"""
import csv
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _ROOT / "scripts/research/ccta_graph_lira_safe_repair"
sys.path.insert(0, str(_SCRIPT_DIR))

from ct28_safety_policy import should_accept


@pytest.mark.parametrize(
    ("proposed", "selected", "compatible", "consistency", "expected"),
    [
        ("PAIR", "PAIR", True, 0.60, True),
        ("JUNCTION", "JUNCTION", True, 0.80, True),
        ("BOTH", "BOTH", True, 1.0, True),
        ("BOTH", "PAIR", False, 1.0, False),
        ("BOTH", "JUNCTION", False, 1.0, False),
        ("PAIR", "NONE", False, 1.0, False),
        ("PAIR", "JUNCTION", True, 1.0, False),
        ("NONE", "NONE", True, 1.0, False),
        ("PAIR", "PAIR", True, 0.59, False),
        ("PAIR", "PAIR", False, 1.0, False),
    ],
)
def test_acceptance_gate(proposed, selected, compatible, consistency, expected):
    assert should_accept(proposed, selected, compatible, consistency, 0.60) is expected


@pytest.mark.parametrize(
    ("relation", "selected", "consistency", "threshold"),
    [
        ("UNKNOWN", "PAIR", 1.0, 0.6),
        ("PAIR", "UNKNOWN", 1.0, 0.6),
        ("PAIR", "PAIR", -0.1, 0.6),
        ("PAIR", "PAIR", 1.1, 0.6),
        ("PAIR", "PAIR", 1.0, 1.1),
    ],
)
def test_invalid_policy_inputs_fail_closed(relation, selected, consistency, threshold):
    with pytest.raises(ValueError):
        should_accept(relation, selected, True, consistency, threshold)


def test_selected_k10_15_validation_does_not_depend_on_missing_compatibility_gate():
    path = (
        _ROOT
        / "artifacts/varvara/ct28_graph_lira/selected_validation/"
        "graph_lira_predictions_val.csv"
    )
    with path.open(encoding="utf-8", newline="") as file:
        predictions = list(csv.DictReader(file))
    assert len(predictions) == 736
    for row in predictions:
        selected = row["selected_relation_before_consistency"]
        proposed = row["tau_relation"]
        accepted = row["accepted"] == "1"
        consistency = float(row["combined_consistency"])
        assert proposed == selected, (row["model"], row["scene_id"])
        assert accepted == should_accept(proposed, selected, True, consistency, 0.60)
    ct = [row for row in predictions if row["model"] == "ct_conditioned"]
    assert sum(row["accepted"] == "1" for row in ct) == 52
    assert sum(row["false_scene"] == "1" for row in ct) == 0
