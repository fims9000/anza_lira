import numpy as np
import pytest
import torch

from models.azconv_affinity import LOCAL8_OFFSETS
from path_completion.maxmin import maxmin_closure_reference, maxmin_closure_torch
from path_completion.widest_path import widest_path
from path_completion.oracle import freeze_train_geometry
from synthetic.affinity_targets import build_affinity_targets
from synthetic.crossing_trace_bench_v5 import generate_sample_v5


def _horizontal_relation(width: int, *, broken_edge: int | None = None) -> np.ndarray:
    relation = np.zeros((8, 1, width), dtype=np.float32)
    right = LOCAL8_OFFSETS.index((1, 0))
    left = LOCAL8_OFFSETS.index((-1, 0))
    for x in range(width - 1):
        if x == broken_edge:
            continue
        relation[right, 0, x] = 1.0
        relation[left, 0, x + 1] = 1.0
    return relation


def test_positive_chain_closes_without_decay() -> None:
    seed = np.asarray([[1, 0, 0, 0, 1]], dtype=np.float32)
    result, _steps = maxmin_closure_reference(seed, _horizontal_relation(5))
    assert np.array_equal(result, np.ones_like(seed))


def test_broken_chain_blocks_wrong_connection() -> None:
    seed = np.asarray([[1, 0, 0, 0, 0]], dtype=np.float32)
    result, _steps = maxmin_closure_reference(seed, _horizontal_relation(5, broken_edge=1))
    assert np.array_equal(result, np.asarray([[1, 1, 0, 0, 0]], dtype=np.float32))


@pytest.mark.parametrize("index", [0, 128, 300, 301])
def test_torch_optimized_equals_reference_on_generator(index: int) -> None:
    sample = generate_sample_v5("validation", index, image_size=32)
    relation = build_affinity_targets(sample, LOCAL8_OFFSETS)["affinity_positive"].astype(np.float32)
    seed = np.asarray(sample["visible_fault_mask"], dtype=np.float32)
    reference, reference_steps = maxmin_closure_reference(seed, relation)
    optimized, optimized_steps = maxmin_closure_torch(
        torch.from_numpy(seed[None]), torch.from_numpy(relation[None])
    )
    assert optimized_steps == reference_steps
    assert np.array_equal(optimized[0, 0].numpy(), reference)


def test_x_crossing_and_two_paths_allow_simultaneous_support() -> None:
    sample = generate_sample_v5("validation", 300, image_size=64)
    relation = build_affinity_targets(sample, LOCAL8_OFFSETS)["affinity_positive"].astype(np.float32)
    latent = np.asarray(sample["latent_fault_mask"], dtype=bool)
    seed = np.zeros_like(latent, dtype=np.float32)
    points = np.argwhere(latent)
    seed[tuple(points[0])] = 1.0
    seed[tuple(points[-1])] = 0.8
    result, _ = maxmin_closure_torch(torch.from_numpy(seed[None]), torch.from_numpy(relation[None]))
    assert torch.isfinite(result).all()
    assert float(result.max()) == 1.0
    assert np.count_nonzero(result.numpy()) > 2


def test_widest_path_uses_bottleneck_and_shorter_tie_break() -> None:
    relation = _horizontal_relation(5)
    relation[:, 0, :] *= 0.8
    score, path = widest_path(relation, (0, 0), (0, 4))
    assert score == pytest.approx(0.8)
    assert path == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    broken_score, broken_path = widest_path(_horizontal_relation(5, broken_edge=2), (0, 0), (0, 4))
    assert broken_score == 0.0 and broken_path == ()


def test_train_only_geometry_freeze_precedes_validation() -> None:
    frozen = freeze_train_geometry()
    assert frozen["d_max_px"] == 18
    assert frozen["path_radius_px"] == 3
    assert frozen["validation_accessed_for_freeze"] is False
