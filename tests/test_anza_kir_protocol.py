from __future__ import annotations

import numpy as np
import pytest

from anza_kir.benchmark import POOL_SIZES, generate_sample
from anza_kir.evaluation import pair_summary
from anza_kir.protocol import protocol, protocol_hash


def test_protocol_freezes_required_pool_and_locks():
    value = protocol()
    assert sum(POOL_SIZES.values()) >= 50_000
    assert value["hard_mining"]["valid_pair_error"] == [0.10, 0.40]
    assert all(value["locks"].values())
    assert protocol_hash(value) == protocol_hash(value)


def test_new_stream_is_deterministic_and_confirm_locked():
    first = generate_sample("mine-dev", 17); second = generate_sample("mine-dev", 17)
    assert np.array_equal(first["image"], second["image"])
    with pytest.raises(PermissionError): generate_sample("mine-confirm", 0)


def test_pair_metric_is_threshold_free_and_task_stratified():
    rows = [
        {"target_score": 0.8, "distractor_score": 0.2, "margin": 0.6, "pair_error": 0, "mechanism_task": "a", "index": 0},
        {"target_score": 0.3, "distractor_score": 0.4, "margin": -0.1, "pair_error": 1, "mechanism_task": "b", "index": 1},
    ]
    summary = pair_summary(rows)
    assert summary["pair_error"] == 0.5
    assert set(summary["per_task"]) == {"a", "b"}
