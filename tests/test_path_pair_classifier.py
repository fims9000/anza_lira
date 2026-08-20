import numpy as np
import torch

from path_completion.pair_classifier import (
    EndpointPairClassifier,
    PAIR_PROTOCOL,
    oriented_pair_crop,
    pair_arrays,
    run_pair_classifier,
)
from path_completion.oracle import freeze_train_geometry
from path_completion.widest_path import candidate_endpoint_pairs
from synthetic.crossing_trace_bench_v5 import generate_sample_v5


def test_oriented_pair_crop_is_deterministic_and_marks_endpoints():
    sample = generate_sample_v5("train", 0)
    frozen = freeze_train_geometry()
    pair = candidate_endpoint_pairs(sample["visible_fault_mask"], d_min=3, d_max=frozen["d_max_px"])[0]
    first = oriented_pair_crop(sample, pair)
    second = oriented_pair_crop(sample, pair)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (6, 33, 49)
    assert first[4].max() > 0.9
    assert np.isfinite(first).all()


def test_pair_arrays_are_balanced_matched_and_split_disjoint():
    train_x, train_y, train_groups = pair_arrays("train", range(2), d_max=18, augment_train=False)
    validation_x, validation_y, validation_groups = pair_arrays("validation", range(2), d_max=18)
    assert train_x.shape == validation_x.shape == (2, 2, 6, 33, 49)
    assert np.array_equal(train_y, np.asarray([[1, 0], [1, 0]], dtype=np.float32))
    assert not set(train_groups.tolist()) & set(validation_groups.tolist())


def test_classifier_forward_and_one_epoch_smoke_are_finite():
    model = EndpointPairClassifier()
    assert model(torch.randn(3, 6, 33, 49)).shape == (3,)
    result, checkpoint = run_pair_classifier(device="cpu", epochs=1, pair_count=4)
    assert result["balanced_pairs"] is True
    assert result["pair_disjoint"] is True
    assert result["confirm_v5_samples_opened"] == result["test_v5_samples_opened"] == 0
    assert np.isfinite(result["validation_metrics"]["auroc"])
    assert checkpoint["protocol_sha256"] == result["protocol_sha256"]


def test_protocol_has_required_context_and_locked_streams():
    assert PAIR_PROTOCOL["local_encoder_receptive_field_px"] >= 9
    assert PAIR_PROTOCOL["validation_auroc_gate"] == 0.85
    assert PAIR_PROTOCOL["confirm"] == PAIR_PROTOCOL["test"] == "LOCKED_UNOPENED"
    assert PAIR_PROTOCOL["cracks_expert"] == "FORBIDDEN"
