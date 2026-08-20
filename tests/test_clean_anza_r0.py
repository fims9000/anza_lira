import torch

from cracks_experiment.clean_anza_r0 import R0_PROTOCOL, audit_r0_reuse_contract, clean_r0_specs
from cracks_experiment.matrix import CRACKSRunSpec, SETTING_A_PROTOCOL
from cracks_experiment.training import build_real_model
from models.azconv_affinity import IndependentFuzzyAZConv2d


def test_r0_has_exactly_three_clean_seeds_and_frozen_real_protocol() -> None:
    specs = clean_r0_specs()
    assert [spec.seed for spec in specs] == [41, 42, 43]
    assert {spec.model for spec in specs} == {"clean_anza"}
    assert R0_PROTOCOL["epochs"] == SETTING_A_PROTOCOL["epochs"]
    assert R0_PROTOCOL["real_loss"] == SETTING_A_PROTOCOL["real_loss"]
    assert R0_PROTOCOL["threshold_candidates"] == SETTING_A_PROTOCOL["threshold_candidates"]


def test_clean_anza_real_model_is_seed_matched_to_legacy_v1_shared_weights() -> None:
    torch.manual_seed(42)
    legacy = build_real_model(CRACKSRunSpec("v1_test", "anza_v1", 42))
    torch.manual_seed(42)
    clean = build_real_model(clean_r0_specs()[1])
    assert any(isinstance(module, IndependentFuzzyAZConv2d) for module in clean.modules())
    legacy_state = legacy.state_dict()
    clean_state = clean.state_dict()
    assert legacy_state.keys() == clean_state.keys()
    assert all(torch.equal(legacy_state[key], clean_state[key]) for key in legacy_state)


def test_reuse_contract_verifies_frozen_unet_and_v1_without_expert_access() -> None:
    result = audit_r0_reuse_contract()
    assert result["status"] == "PASS"
    assert len(result["reused_runs"]) == 6
    assert result["expert_scores_used_for_selection"] is False
    assert result["expert_data_accessed"] is False

