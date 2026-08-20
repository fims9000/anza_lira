import numpy as np

from anza_ks.features import kolmogorov_information_features
from anza_ks_k2.features import information_features, shear_ks_feature_vector
from anza_ks_k2.freeze import freeze_k1_5


def test_cat_information_regression_is_exact() -> None:
    rng = np.random.default_rng(15)
    patch = rng.normal(size=(17, 17))
    np.testing.assert_array_equal(information_features(patch, "cat"), kolmogorov_information_features(patch))


def test_shear_ks_has_frozen_width_and_finite_values() -> None:
    patch = np.arange(289, dtype=np.float64).reshape(17, 17)
    values = shear_ks_feature_vector(patch)
    assert values.shape == (104,)
    assert np.isfinite(values).all()


def test_information_dynamics_is_validated() -> None:
    try:
        information_features(np.zeros((17, 17)), "rotation")
    except ValueError as error:
        assert "dynamics" in str(error)
    else:
        raise AssertionError("unknown dynamics must fail closed")


def test_k1_5_freeze_preserves_parent_and_lock() -> None:
    receipt = freeze_k1_5()
    assert receipt["parent_package_sha256"] == "cd4de1fb01551e616acab9270f984726a8c92264892b2a98559d68001a56df67"
    assert receipt["old_readouts_retrained"] is False
    assert receipt["confirm_evaluated"] is False
