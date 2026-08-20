import numpy as np
import torch

from connectivity_repair.receptive_field_probe import (
    RECEPTIVE_FIELDS,
    ReceptiveFieldProbe,
    parameter_count,
    probe_arrays,
    run_rf_probe,
)
from synthetic.crossing_trace_bench_v4 import sample_seed_v4
from synthetic.crossing_trace_bench_v5 import generate_sample_v5, sample_seed_v5


def test_v5_is_independent_deterministic_and_test_locked():
    first = generate_sample_v5("validation", 17, image_size=64)
    second = generate_sample_v5("validation", 17, image_size=64)
    np.testing.assert_array_equal(first["image"], second["image"])
    assert sample_seed_v5("validation", 17) != sample_seed_v4("validation", 17)
    try:
        generate_sample_v5("test", 0, image_size=32)
    except PermissionError as error:
        assert "LOCKED" in str(error)
    else:
        raise AssertionError("v5 test was opened")


def test_rf_models_have_equal_capacity_and_declared_effective_field():
    assert len({parameter_count(value) for value in RECEPTIVE_FIELDS}) == 1
    for receptive_field in RECEPTIVE_FIELDS:
        model = ReceptiveFieldProbe(receptive_field)
        assert 1 + 2 * model.steps == receptive_field
        assert model.shared_update.dilation == (1, 1)
        assert model(torch.randn(2, 3, 64, 64)).shape == (2,)


def test_probe_arrays_are_balanced_and_pair_disjoint():
    _train_x, train_y, train_groups = probe_arrays("train", range(0, 4), image_size=64, crop_size=48)
    _val_x, val_y, val_groups = probe_arrays("validation", range(96, 100), image_size=64, crop_size=48)
    assert train_y.mean() == val_y.mean() == 0.5
    assert not set(train_groups.tolist()) & set(val_groups.tolist())
    assert np.all(train_groups[0::2] == train_groups[1::2])


def test_one_epoch_probe_smoke_is_finite_and_fail_closed():
    result = run_rf_probe(device="cpu", epochs=1)
    assert result["pair_disjoint"] is True
    assert result["test_v5_samples_opened"] == 0
    assert all(np.isfinite(row["validation_auroc"]) for row in result["rows"])
