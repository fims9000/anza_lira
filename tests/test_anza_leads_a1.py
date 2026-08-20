from __future__ import annotations

import torch
from pathlib import Path

from anza_leads.evaluation import select_threshold
from anza_leads.model import build_leads_model
from anza_leads.protocol import PROTOCOL
from anza_leads import training


def test_threshold_selection_obeys_frozen_precision_constraint() -> None:
    curve = [
        {"threshold": 0.3, "precision": 0.89, "cldice": 0.99, "dice": 0.9},
        {"threshold": 0.4, "precision": 0.91, "cldice": 0.80, "dice": 0.8},
        {"threshold": 0.5, "precision": 0.95, "cldice": 0.75, "dice": 0.7},
    ]
    selected = select_threshold(curve)
    assert selected["selected_threshold"] == 0.4
    assert selected["constraint_feasible"]
    assert selected["precision_target"] == 0.90


def test_threshold_infeasibility_is_reported_not_hidden() -> None:
    curve = [
        {"threshold": 0.3, "precision": 0.70, "cldice": 0.9, "dice": 0.8},
        {"threshold": 0.4, "precision": 0.80, "cldice": 0.7, "dice": 0.7},
    ]
    selected = select_threshold(curve)
    assert selected["selected_threshold"] == 0.4
    assert not selected["constraint_feasible"]


def test_model_outputs_and_gradients_are_finite() -> None:
    for variant in ("L0_backbone", "L1_isotropic", "L2_generic_aniso", "L3_anza_hs"):
        model = build_leads_model(variant)
        output = model(torch.randn(1, 3, 64, 64), return_aux=True)
        loss = output["visible_logits"].square().mean() + sum(value.square().mean() for value in output["orientation_logits"])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_a1_budget_and_locks_are_frozen() -> None:
    assert PROTOCOL["training"]["active_seed"] == 41
    assert PROTOCOL["training"]["active_fraction"] == 0.10
    assert PROTOCOL["training"]["epochs"] == 20
    assert PROTOCOL["training"]["variant_specific_augmentation"] is False
    assert not any(PROTOCOL["locks"].values())


def test_checkpoint_reload_skips_only_deterministic_meshgrid_views(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(training, "run_hash", lambda _variant: "testhash")
    monkeypatch.setattr(training, "protocol_hash", lambda: "protocolhash")
    source = build_leads_model("L1_isotropic")
    path = Path(tmp_path) / "checkpoint.pt"
    torch.save({
        "variant": "L1_isotropic", "run_hash": "testhash", "protocol_sha256": "protocolhash",
        "seed": 41, "label_fraction": 0.10, "expert_data_accessed": False,
        "development_data_accessed": False, "model_state": source.state_dict(),
    }, path)
    target = build_leads_model("L1_isotropic")
    training.load_checkpoint(path, "L1_isotropic", target)
    x = torch.randn(1, 3, 32, 32)
    source.eval(); target.eval()
    with torch.no_grad():
        assert torch.equal(source(x), target(x))
