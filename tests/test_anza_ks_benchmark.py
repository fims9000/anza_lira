import numpy as np
import pytest

from anza_ks.benchmark.matched_generator import SPLIT_SEEDS, SPLIT_SIZES, TASKS, benchmark_manifest, generate_pair
from anza_ks.benchmark.static_signature import STATIC_SIGNATURE_VERSION, static_signature


def test_benchmark_contract_and_split_seeds_are_disjoint():
    manifest = benchmark_manifest()
    assert manifest["version"] == "ANZA_KS_DYNAMICS_MATCHED_V1"
    assert len(TASKS) == 5
    assert SPLIT_SIZES == {"train": 2048, "dev": 1024, "confirm": 2048}
    assert len(set(SPLIT_SEEDS.values())) == 3
    assert manifest["anza_ks_used_for_generation"] is False
    assert STATIC_SIGNATURE_VERSION == "ANZA_KS_STATIC_SIGNATURE_V1"


@pytest.mark.parametrize("task", TASKS)
def test_pairs_are_deterministic_observable_and_static_matched(task):
    first = generate_pair(task, "dev", 3)
    second = generate_pair(task, "dev", 3)
    assert np.array_equal(first["positive"], second["positive"])
    assert np.array_equal(first["negative"], second["negative"])
    assert not first["pixel_equal"] and first["l2_difference"] > 1e-3
    difference = np.linalg.norm(static_signature(first["positive"]) - static_signature(first["negative"]))
    assert difference <= 1e-8


def test_confirm_samples_are_access_locked():
    with pytest.raises(PermissionError):
        generate_pair(TASKS[0], "confirm", 0)
