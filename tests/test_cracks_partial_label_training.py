import copy

import pytest
import torch

from cracks_experiment.partial_label_evaluation import _calibration, _select_threshold
from cracks_experiment.partial_label_training import (
    T1_PROTOCOL,
    _model,
    load_t1_checkpoint,
    t1_matrix,
    t1_protocol_hash,
)


def test_t1_matrix_is_exactly_two_models_three_seeds() -> None:
    matrix = t1_matrix()
    assert [(row.model, row.seed) for row in matrix] == [
        ("unet", 41), ("unet", 42), ("unet", 43),
        ("anza_v1", 41), ("anza_v1", 42), ("anza_v1", 43),
    ]
    assert len({row.run_hash for row in matrix}) == 6
    assert set(T1_PROTOCOL["training_annotators"]).isdisjoint(T1_PROTOCOL["heldout_annotators"])
    assert "expert" not in T1_PROTOCOL["training_annotators"]
    assert "expert" not in T1_PROTOCOL["heldout_annotators"]


def test_t1_hash_changes_with_target_semantics() -> None:
    original = copy.deepcopy(T1_PROTOCOL["target_semantics"])
    before = t1_protocol_hash()
    try:
        T1_PROTOCOL["target_semantics"]["green"] = [1.0, 0.75]
        assert t1_protocol_hash() != before
    finally:
        T1_PROTOCOL["target_semantics"] = original
    assert t1_protocol_hash() == before


def test_t1_checkpoint_rejects_hash_and_expert_drift(tmp_path) -> None:
    spec = t1_matrix()[0]
    model = _model(spec)
    optimizer = torch.optim.Adam(model.parameters())
    payload = {
        "run_hash": "wrong",
        "protocol_sha256": t1_protocol_hash(),
        "expert_scores_used": False,
        "expert_data_accessed": False,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_t1_checkpoint(path, spec, model)
    payload["run_hash"] = spec.run_hash
    payload["expert_data_accessed"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_t1_checkpoint(path, spec, model)


def test_partial_threshold_is_section_macro_and_ties_choose_lower() -> None:
    rows = []
    for threshold, values in ((0.1, (0.5, 0.7)), (0.2, (0.5, 0.7))):
        for section, dice in enumerate(values):
            rows.append({"threshold": threshold, "section_id": section, "dice": dice, "recall": 0.8})
    result = _select_threshold(rows)
    assert result["selected_threshold"] == 0.1
    assert result["selection_metric"].startswith("macro_section")


def test_weighted_calibration_ignores_white() -> None:
    probability = torch.tensor([[0.9, 0.1, 1.0]], dtype=torch.float32).numpy()
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).numpy()
    weight = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32).numpy()
    brier, ece = _calibration(probability, target, weight)
    assert brier == pytest.approx(0.01, abs=1e-6)
    assert ece == pytest.approx(0.1, abs=1e-6)
