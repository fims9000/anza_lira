from anza2_experiment.synthetic_replacement import protocol_payload


def test_phase2b_protocol_preserves_phase2a_and_freezes_independent_primary() -> None:
    protocol = protocol_payload()
    assert protocol["phase2a_metrics_sha256"] == "04b35a97c830b682f682084498673daf280e1c81dad407e850be199e8e15e383"
    assert protocol["replacement_confirm_seed_base"] == 630_000_000
    assert protocol["primary_metric"].startswith("junction branch recall delta")
    assert protocol["training_performed"] is False
    assert protocol["expert_data_accessed"] is False
