from __future__ import annotations

import numpy as np
import pytest

from anza_tracegraph.data_v2.generator import BUILDERS, generate_scene
from anza_tracegraph.data_v2.strata import NEGATIVE_STRATA, POSITIVE_STRATA, SPLIT_SEEDS, STRATA
from anza_tracegraph.data_v2.validator import validate_generator


def test_every_stratum_has_a_distinct_explicit_constructor():
    assert set(BUILDERS) == set(STRATA)
    assert len(set(BUILDERS.values())) == len(STRATA)


def test_names_determine_relation_polarity_without_hidden_flag():
    for name in STRATA:
        scene = generate_scene("calibration", STRATA.index(name))
        assert scene["truth"]["has_valid_continuation"] == (name in POSITIVE_STRATA)
        assert (scene["truth"]["destination_branch"] is not None) == (name in POSITIVE_STRATA)
    assert all(name not in POSITIVE_STRATA for name in NEGATIVE_STRATA)


def test_generator_semantic_contract_and_no_truth_in_public_input():
    assert validate_generator()["validator"] == "PASS"
    scene = generate_scene("development", 0)
    assert scene["input"]["model_input"].shape == (3, 96, 96)
    assert set(scene["input"]).isdisjoint({"destination_branch", "destination_id", "true_path", "target_endpoint", "has_valid_continuation"})


def test_splits_are_disjoint_and_confirm_is_inaccessible():
    assert len(set(SPLIT_SEEDS.values())) == 3
    assert not np.array_equal(generate_scene("calibration", 17)["input"]["model_input"], generate_scene("development", 17)["input"]["model_input"])
    with pytest.raises(PermissionError): generate_scene("confirm", 17)


def test_positive_strata_have_exactly_one_primary_truth_branch():
    for name in POSITIVE_STRATA:
        truth = generate_scene("calibration", STRATA.index(name))["truth"]
        assert truth["destination_branch"] is not None
        assert truth["destination_branch"].ndim == 2


def test_x_t_y_weak_and_multiple_have_declared_geometry():
    rows = {name: generate_scene("calibration", STRATA.index(name))["truth"] for name in ("x_crossing_correct", "t_junction_continue", "y_junction_continue", "weak_branch_continue", "multiple_plausible_correct")}
    assert rows["x_crossing_correct"]["topology"] == "x_crossing"
    assert rows["t_junction_continue"]["topology"] == "t_junction" and len(rows["t_junction_continue"]["distractor_branches"]) == 1
    assert rows["y_junction_continue"]["topology"] == "y_junction" and len(rows["y_junction_continue"]["distractor_branches"]) == 1
    assert rows["weak_branch_continue"]["destination_signal"] < max(rows["weak_branch_continue"]["competitor_signals"])
    assert len(rows["multiple_plausible_correct"]["distractor_branches"]) >= 1
