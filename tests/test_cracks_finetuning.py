import pytest

from cracks_experiment.finetuning import FOLDS, SETTING_B_PROTOCOL, setting_b_sources, verify_setting_a_complete


def test_setting_b_matrix_and_folds_are_frozen() -> None:
    assert [spec.model for spec in setting_b_sources()] == [
        "unet", "deformable_unet", "anza_v1", "anza_v2b"
    ]
    assert len(FOLDS["folds"]) == SETTING_B_PROTOCOL["fold_count"] == 5
    assert all(
        (len(fold["train"]), len(fold["validation"]), len(fold["test"])) == (28, 4, 8)
        for fold in FOLDS["folds"]
    )


def test_setting_b_stays_locked_until_setting_a_complete(tmp_path) -> None:
    with pytest.raises(PermissionError, match="receipt missing"):
        verify_setting_a_complete(tmp_path / "train", tmp_path / "expert")
