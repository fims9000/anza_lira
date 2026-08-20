from __future__ import annotations

from cracks_experiment.matrix import setting_a_matrix
from cracks_experiment.training import run_setting_a_training


def test_setting_a_tiny_cpu_train_reload_and_skip(tmp_path) -> None:
    spec = next(run for run in setting_a_matrix() if run.run_id == "unet_s42")
    first = run_setting_a_training(
        spec,
        tmp_path,
        epochs=1,
        max_train_sections=1,
        device="cpu",
    )
    second = run_setting_a_training(
        spec,
        tmp_path,
        epochs=1,
        max_train_sections=1,
        device="cpu",
    )
    assert first["status"] == "COMPLETE" and first["checkpoint_reload"] == "PASS"
    assert first["expert_scores_used"] is False
    assert second["action"] == "SKIP"
