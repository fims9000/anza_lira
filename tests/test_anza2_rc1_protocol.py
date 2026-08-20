import numpy as np
import torch

from anza2_experiment.learned_affinity import LearnedAffinityModel, _batch
import anza2_experiment.rc1_membership_repair as rc1
from anza2_experiment.rc1_membership_repair import (
    CONFIGS,
    configure_beta_only,
    configure_membership_only,
    protocol_payload,
)
from models.anza2.losses import rc1_membership_loss


def test_rc1_protocol_freezes_exactly_two_variants_and_all_data_locks():
    protocol = protocol_payload()
    assert CONFIGS == {"M-A": 0.25, "M-B": 0.50}
    assert protocol["lambda_count"] == 0.25
    assert protocol["epochs"] == 5
    assert protocol["minimum_tpr_delta"] == 0.08
    assert protocol["confirm_opened"] is False
    assert protocol["cracks_data_accessed"] is False
    assert protocol["expert_data_accessed"] is False


def test_only_membership_head_receives_rc1_gradients():
    model = LearnedAffinityModel(initial_beta=0.05)
    names = configure_membership_only(model)
    images, targets = _batch("train", [0], 64, torch.device("cpu"))
    output = model(images, use_anza=True)
    theta = targets["theta"].float()
    target_orientation = torch.stack((torch.cos(2 * theta), torch.sin(2 * theta)), dim=2)
    loss, _ = rc1_membership_loss(
        output["field"].orientation, output["field"].membership,
        target_orientation, targets["theta_valid"], targets["mode_count"].float(),
        lambda_bg=0.25,
    )
    loss.backward()
    assert names == ["field.membership_head.weight", "field.membership_head.bias"]
    assert all(parameter.grad is not None for parameter in model.field.membership_head.parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if name not in names)


def test_membership_step_preserves_geometry_generic_and_beta_bitwise():
    model = LearnedAffinityModel(initial_beta=0.05)
    configure_membership_only(model)
    frozen = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("field.membership_head.")
    }
    membership_before = model.field.membership_head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.field.membership_head.parameters(), lr=0.001)
    images, targets = _batch("train", [0], 64, torch.device("cpu"))
    output = model(images, use_anza=True); theta = targets["theta"].float()
    target_orientation = torch.stack((torch.cos(2 * theta), torch.sin(2 * theta)), dim=2)
    loss, _ = rc1_membership_loss(
        output["field"].orientation, output["field"].membership,
        target_orientation, targets["theta_valid"], targets["mode_count"].float(), lambda_bg=0.25,
    )
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(membership_before, model.field.membership_head.weight.detach())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())


def test_beta_only_configuration_changes_trainable_surface_only():
    model = LearnedAffinityModel(initial_beta=0.05)
    names = configure_beta_only(model)
    assert names == ["combiner.beta_raw"]
    assert model.combiner.beta_raw.requires_grad
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name != "combiner.beta_raw")


def test_beta_refit_changes_only_beta(monkeypatch):
    model = LearnedAffinityModel(initial_beta=0.05)
    frozen = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name != "combiner.beta_raw"
    }
    beta_before = model.combiner.beta_raw.detach().clone()
    record = {
        "index": 0,
        "positive_generic_logits": np.zeros(8, dtype=np.float32),
        "negative_generic_logits": np.zeros(8, dtype=np.float32),
        "positive_prior_logits": np.ones(8, dtype=np.float32),
        "negative_prior_logits": -np.ones(8, dtype=np.float32),
    }
    monkeypatch.setattr(rc1, "_fixed_edge_records", lambda *args, **kwargs: [record])
    result = rc1.fit_beta_train_only(model, seed=41, device=torch.device("cpu"))
    assert result["development_used_for_fit"] is False
    assert not torch.equal(beta_before, model.combiner.beta_raw.detach())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())


def test_confirm_has_no_callable_entrypoint_before_development_gate():
    assert not hasattr(rc1, "run_confirm")
    assert protocol_payload()["confirm_opened"] is False
