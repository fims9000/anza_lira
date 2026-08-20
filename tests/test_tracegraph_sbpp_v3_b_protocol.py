from __future__ import annotations

import inspect

import numpy as np
import pytest

from anza_tracegraph.data_v2.generator import generate_scene
from anza_tracegraph.ports_v3_b.clustering import cluster_branches
from anza_tracegraph.ports_v3_b.repair_data import REPAIR_CALIBRATION_SEED, generate_repair_scene
from anza_tracegraph.ports_v3_b.soft_branches import extract_soft_branches


def test_repair_calibration_is_disjoint_from_old_splits():
    assert REPAIR_CALIBRATION_SEED not in {5_201_000_000, 5_211_000_000, 5_221_000_000}
    assert not np.array_equal(generate_repair_scene(13)["input"]["model_input"], generate_scene("calibration", 13)["input"]["model_input"])


def test_truth_is_not_an_extraction_or_clustering_argument():
    assert "truth" not in inspect.signature(extract_soft_branches).parameters
    assert "truth" not in inspect.signature(cluster_branches).parameters


def test_old_development_and_confirm_locks_are_unchanged():
    with pytest.raises(PermissionError): generate_scene("confirm", 0)
    scene = generate_scene("development", 0)
    assert scene["input"]["split"] == "development"
