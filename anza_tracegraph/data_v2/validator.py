"""Semantic checks for TRACEGRAPH_RELATION_V2."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .generator import BUILDERS, generate_scene
from .strata import NEGATIVE_STRATA, POSITIVE_STRATA, SPLIT_SEEDS, STRATA


def _minimum_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(first[:, None] - second[None], axis=2)))


def validate_generator() -> dict[str, Any]:
    samples = {name: generate_scene("calibration", STRATA.index(name)) for name in STRATA}
    checks = {
        "dedicated_constructor_per_stratum": set(BUILDERS) == set(STRATA) and len(set(BUILDERS.values())) == len(STRATA),
        "negative_names_are_none": all(not samples[name]["truth"]["has_valid_continuation"] and samples[name]["truth"]["destination_branch"] is None for name in NEGATIVE_STRATA),
        "positive_names_have_one_primary": all(samples[name]["truth"]["has_valid_continuation"] and samples[name]["truth"]["destination_branch"] is not None for name in POSITIVE_STRATA),
        "x_has_crossing": _minimum_distance(samples["x_crossing_correct"]["truth"]["destination_branch"], samples["x_crossing_correct"]["truth"]["distractor_branches"][0]) <= 2.0,
        "t_has_three_arms": samples["t_junction_continue"]["truth"]["topology"] == "t_junction" and len(samples["t_junction_continue"]["truth"]["distractor_branches"]) == 1,
        "y_has_three_arms": samples["y_junction_continue"]["truth"]["topology"] == "y_junction" and len(samples["y_junction_continue"]["truth"]["distractor_branches"]) == 1,
        "weak_target_is_weaker": samples["weak_branch_continue"]["truth"]["destination_signal"] < max(samples["weak_branch_continue"]["truth"]["competitor_signals"]),
        "multiple_plausible_has_competitor": len(samples["multiple_plausible_correct"]["truth"]["distractor_branches"]) >= 1 and _minimum_distance(samples["multiple_plausible_correct"]["truth"]["destination_branch"], samples["multiple_plausible_correct"]["truth"]["distractor_branches"][0]) <= 6.0,
        "split_seeds_disjoint": len(set(SPLIT_SEEDS.values())) == len(SPLIT_SEEDS),
        "public_input_has_no_truth": all(set(scene["input"]).isdisjoint({"destination_branch", "destination_id", "true_path", "target_endpoint", "has_valid_continuation"}) for scene in samples.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed: raise ValueError(f"TRACEGRAPH_RELATION_V2 invalid: {failed}")
    return {"validator": "PASS", "checks": checks, "strata": len(STRATA)}
