import numpy as np
import torch

from path_completion.learned_confirm import (
    CONFIRM_PROTOCOL,
    load_frozen_pair_classifier,
    pair_gated_completion,
    score_pair,
    straight_shortest_path,
)
from path_completion.pair_classifier import _single_pair
from path_completion.widest_path import EndpointPair
from synthetic.crossing_trace_bench_v5 import generate_sample_v5


def test_straight_shortest_path_is_deterministic_and_endpoint_exact():
    pair = EndpointPair((3, 4), (8, 11), np.hypot(5, 7))
    path = straight_shortest_path(pair)
    assert path[0] == pair.first and path[-1] == pair.second
    assert path == straight_shortest_path(pair)
    assert len(path) == 8


def test_rejected_pair_preserves_visible_mask_exactly():
    sample = generate_sample_v5("validation", 128)
    pair = _single_pair(sample, 18)
    completion = pair_gated_completion(sample, pair, score=0.1, threshold=0.5, path_radius=3)
    np.testing.assert_array_equal(completion, sample["visible_fault_mask"])


def test_accepted_pair_adds_only_and_never_removes_visible_pixels():
    sample = generate_sample_v5("validation", 0)
    pair = _single_pair(sample, 18)
    completion = pair_gated_completion(sample, pair, score=0.9, threshold=0.5, path_radius=3)
    visible = np.asarray(sample["visible_fault_mask"], dtype=bool)
    assert np.all(completion[visible])
    assert completion.sum() > visible.sum()


def test_frozen_classifier_loads_and_scores_without_truth_inputs():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    model, frozen = load_frozen_pair_classifier(root, torch.device("cpu"))
    sample = generate_sample_v5("validation", 0)
    pair = _single_pair(sample, frozen["d_max_px"])
    score = score_pair(model, sample, pair, torch.device("cpu"))
    assert 0.0 <= score <= 1.0


def test_confirm_protocol_is_fail_closed_for_test_and_expert():
    assert CONFIRM_PROTOCOL["test"] == "LOCKED_UNOPENED"
    assert CONFIRM_PROTOCOL["cracks_expert"] == "FORBIDDEN"
    assert CONFIRM_PROTOCOL["gates"]["pair_auroc_min"] == 0.85
