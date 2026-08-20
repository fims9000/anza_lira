import numpy as np
import torch

from anza_ks.benchmark.static_signature import static_signature
from anza_ks.features import dynamic_feature_vector
from anza_ks_k2.dense_features import dense_orientation_features, features_from_patches
from anza_ks_k2.features import shear_ks_feature_vector


def test_dense_numpy_equivalence_all_families() -> None:
    rng = np.random.default_rng(2019)
    patch = rng.normal(size=(17, 17))
    tensor = torch.tensor(patch, dtype=torch.float64)
    expected = {
        "static": static_signature(patch),
        "cat_raw": dynamic_feature_vector(patch, "K1_C_cat_raw"),
        "cat_ks": dynamic_feature_vector(patch, "K1_D_anza_ks"),
        "shear_ks": shear_ks_feature_vector(patch),
    }
    for method, reference in expected.items():
        actual = features_from_patches(tensor, method).detach().numpy()
        np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-6, err_msg=method)


def test_dense_features_have_finite_gradients() -> None:
    patch = torch.randn(2, 17, 17, requires_grad=True)
    total = sum(features_from_patches(patch, method).square().mean() for method in ("static", "shear_ks", "cat_raw", "cat_ks"))
    total.backward()
    assert patch.grad is not None
    assert torch.isfinite(patch.grad).all()


def test_dense_orientation_contract() -> None:
    structural = torch.randn(1, 1, 18, 19)
    values = dense_orientation_features(structural, "cat_raw")
    assert values.shape == (1, 18, 19, 8, 104)
    assert torch.isfinite(values).all()
