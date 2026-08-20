from pathlib import Path

import numpy as np

from lira_final.data.cracks_trace_audit import CrowdTrace
from lira_final.data.natural_gaps import find_natural_gaps
from lira_final.data.splits import build_split_manifest
from lira_final.protocol import HELDOUT_ANNOTATORS, PROTOCOL


def test_final_splits_are_disjoint_and_expert_locked() -> None:
    manifest = build_split_manifest()
    sets = [set(values) for values in manifest["splits"].values()]
    assert all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets)))
    assert "expert" not in HELDOUT_ANNOTATORS
    assert PROTOCOL["locks"]["expert"] and PROTOCOL["locks"]["confirm"]


def test_natural_gap_requires_continuous_annotation_and_visible_context() -> None:
    points = np.asarray([(10.0, float(x)) for x in range(60)], dtype=np.float32)
    trace = CrowdTrace(1, "novice12", "trace", points)
    probability = np.ones((32, 80), dtype=np.float32)
    probability[10, 25:35] = 0.0
    gaps = find_natural_gaps(trace, probability, 0.5)
    assert len(gaps) == 1
    assert gaps[0].start_index == 25 and gaps[0].end_index == 35


def test_short_or_edge_gap_is_not_a_natural_gap() -> None:
    points = np.asarray([(10.0, float(x)) for x in range(40)], dtype=np.float32)
    trace = CrowdTrace(1, "novice12", "trace", points)
    probability = np.ones((32, 60), dtype=np.float32)
    probability[10, 2:12] = 0.0
    probability[10, 20:24] = 0.0
    assert find_natural_gaps(trace, probability, 0.5) == ()

