from cracks_experiment.finetuning import FOLDS
from cracks_experiment.robustness import SETTING_C_PROTOCOL, setting_c_models


def test_setting_c_models_and_neighbor_exclusions_are_frozen() -> None:
    assert [spec.model for spec in setting_c_models()] == ["unet", "anza_v1", "anza_v2b"]
    assert SETTING_C_PROTOCOL["neighbor_guard"] == 2
    for fold in FOLDS["folds"]:
        excluded = set(fold["setting_c_excluded_section_ids"])
        assert set(fold["test"]).issubset(excluded)
        for section in fold["test"]:
            assert {section + delta for delta in range(-2, 3)}.issubset(excluded)
