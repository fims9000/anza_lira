from __future__ import annotations

from cracks_experiment.matrix import SETTING_A_PROTOCOL, setting_a_matrix, setting_a_protocol_hash


def test_setting_a_main_matrix_has_four_models_and_three_seeds() -> None:
    main = [run for run in setting_a_matrix() if run.comparison_family == "main"]
    assert len(main) == 12
    assert {run.seed for run in main} == {41, 42, 43}
    for seed in (41, 42, 43):
        assert {run.model for run in main if run.seed == seed} == {
            "unet",
            "deformable_unet",
            "anza_v1",
            "anza_v2b",
        }


def test_real_protocol_is_fair_and_expert_locked() -> None:
    assert SETTING_A_PROTOCOL["crop_size"] == 256
    assert SETTING_A_PROTOCOL["foreground_crop_probability"] == 0.7
    assert SETTING_A_PROTOCOL["real_loss"] == "bce+dice+0.2*soft_cldice"
    assert SETTING_A_PROTOCOL["expert_scores"] == "LOCKED"
    assert len(setting_a_protocol_hash()) == 16


def test_v2_ablation_matrix_does_not_fake_unused_junction_or_cone_switches() -> None:
    ablations = [run.run_id for run in setting_a_matrix() if run.comparison_family == "ablation"]
    assert ablations == ["v2_no_replay_s42", "v2_no_fuzzy_s42", "v2_no_directional_s42"]
