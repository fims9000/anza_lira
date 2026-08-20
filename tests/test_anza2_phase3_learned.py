import torch

import numpy as np

from anza2_experiment.learned_affinity import LearnedAffinityModel, _metrics, _threshold, canonical_hash, protocol_payload, run_phase3


def test_generic_off_is_exactly_generic_logits():
    torch.manual_seed(1)
    model = LearnedAffinityModel()
    image = torch.randn(2, 3, 17, 17)
    output = model(image, use_anza=False)
    assert torch.equal(output["logits"], output["generic_logits"])


def test_protocol_is_locked_and_expert_free():
    protocol = protocol_payload()
    assert len(canonical_hash(protocol)) == 64
    assert protocol["confirm_open_rule"].startswith("development gate")
    assert protocol["cracks_data_accessed"] is False
    assert protocol["expert_data_accessed"] is False


def test_phase3_smoke_is_small_and_does_not_open_confirm(tmp_path):
    result = run_phase3(tmp_path, mode="smoke", device="cpu")
    assert result["status"] == "SMOKE_PASS"
    assert result["confirm_opened"] is False
    assert result["cracks_data_accessed"] is False
    assert result["expert_data_accessed"] is False
    assert {row["seed"] for row in result["seed_metrics"]} == {41}


def test_threshold_never_exceeds_inclusive_fpr_budget_with_ties():
    rows = [{
        "positive_scores": np.array([0.9, 0.6], dtype=np.float32),
        "negative_scores": np.array([0.9] * 6 + [0.8] * 6 + [0.1] * 88, dtype=np.float32),
    }]
    threshold = _threshold(rows, target_fpr=0.05)
    assert _metrics(rows, threshold)["fpr"] <= 0.05
