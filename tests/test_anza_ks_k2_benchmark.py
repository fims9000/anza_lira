import numpy as np
import pytest

from anza_ks_k2.benchmark import SPLIT_SIZES, generate_sample


def test_k2_train_is_exact_half_mixture_and_deterministic() -> None:
    a = generate_sample("train", 7)
    b = generate_sample("train", 7)
    assert a["domain"] != "natural"
    np.testing.assert_array_equal(a["image"], b["image"])
    assert generate_sample("train", 8)["domain"] == "natural"


def test_k2_scene_contract() -> None:
    for split in ("dev-natural", "dev-mechanism"):
        sample = generate_sample(split, 0)
        assert sample["image"].shape == (3, 96, 96)
        assert sample["target"].shape == (96, 96)
        assert sample["distractor"].shape == (96, 96)
        assert sample["orientation_bank"].shape == (8, 96, 96)
        assert sample["target"].sum() > 0
        assert np.isfinite(sample["image"]).all()


def test_k2_confirm_is_locked() -> None:
    assert SPLIT_SIZES["confirm"] == 2048
    with pytest.raises(PermissionError):
        generate_sample("confirm", 0)
