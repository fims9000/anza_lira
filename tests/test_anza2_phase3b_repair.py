from anza2_experiment.learned_affinity_repair import PHASE3_V1_PROTOCOL_SHA256, protocol_payload


def test_phase3b_is_one_causal_repair_with_locked_confirm():
    protocol = protocol_payload()
    assert protocol["parent_phase3_protocol_sha256"] == PHASE3_V1_PROTOCOL_SHA256
    assert protocol["primary_comparison"].startswith("same checkpoint")
    assert protocol["confirm_open_rule"].startswith("Phase-3B development gate")
    assert protocol["cracks_data_accessed"] is False
    assert protocol["expert_data_accessed"] is False
