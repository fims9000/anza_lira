"""Numerical invariants for the new 3D ANZA-style physical kernel.

Synthetic tensors only; this does not assert CCTA model performance.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.azconv3d import (
    AZConv3d,AZConv3dConfig,quadratic_metric,pair_kernel,count_trainable
)


def _direction(values):
    t=torch.tensor(values,dtype=torch.float64)
    return t.reshape(1,1,3,1,1,1)


def test_uniaxial_metric_has_unit_determinant():
    h=torch.tensor([[[[[math.log(2.0)]]]]],dtype=torch.float64)
    u=_direction([1,0,0])
    axes=[
        torch.tensor([1.,0.,0.],dtype=torch.float64),
        torch.tensor([0.,1.,0.],dtype=torch.float64),
        torch.tensor([0.,0.,1.],dtype=torch.float64),
    ]
    diag=[quadratic_metric(u,h,d).item() for d in axes]
    assert diag==pytest.approx([4.,.5,.5])
    assert math.prod(diag)==pytest.approx(1.,abs=1e-12)


def test_pairwise_kernel_is_symmetric_on_swapped_endpoints():
    u1=_direction([1.,0.,0.])
    u2=_direction([0.,1.,0.])
    h1=torch.ones((1,1,1,1,1),dtype=torch.float64)*.6
    h2=-h1*.8
    a=torch.ones_like(h1)*.2
    b=torch.ones_like(h1)*.7
    d=torch.tensor([1.5,2.,-3.],dtype=torch.float64)
    first=pair_kernel(a,b,u1,u2,h1,h2,d,beta=.2)
    swapped=pair_kernel(b,a,u2,u1,h2,h1,-d,beta=.2)
    assert torch.allclose(first,swapped,atol=1e-14,rtol=0)


def test_3d_operator_output_and_backprop():
    torch.manual_seed(17)
    model=AZConv3d(2,3,AZConv3dConfig(
        modes=2,kernel_size=3,value_channels=3,
        beta=.14,max_hyperbolicity=.7,
    ))
    x=torch.randn((1,2,5,6,5),requires_grad=True)
    basis=torch.diag(torch.tensor([.6,.8,1.4]))
    output=model(x,basis)
    assert output.shape==(1,3,5,6,5)
    assert torch.isfinite(output).all()
    loss=output.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for parameter in (model.membership.weight,
                      model.orientation.weight,
                      model.hyperbolicity.weight,
                      model.values.weight,
                      model.mix.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    assert count_trainable(model)>0


def test_isotropic_fuzzy_ablation_and_spacing():
    torch.manual_seed(2)
    model=AZConv3d(1,2,AZConv3dConfig(
        modes=2,kernel_size=1,value_channels=2,
        anisotropy=False,fuzzy=False,
    ))
    input=torch.randn(1,1,3,4,5)
    output=model(input,torch.eye(3))
    assert output.shape==(1,2,3,4,5)
    assert torch.isfinite(output).all()
    with pytest.raises(ValueError,match="Singular"):
        model(input,torch.zeros(3,3))


def test_invalid_settings_and_shape_rejected():
    with pytest.raises(ValueError,match="odd kernel"):
        AZConv3dConfig(kernel_size=2)
    model=AZConv3d(1,1)
    with pytest.raises(ValueError,match="B,C,D,H,W"):
        model(torch.rand(2,4,4,4))
    with pytest.raises(ValueError,match="nonfinite"):
        x=torch.ones(1,1,4,4,4)
        x[0,0,0,0,0]=float("nan")
        model(x)


def test_contrastive_anisotropic_operator_annihilates_constant_feature_field():
    torch.manual_seed(128)
    model=AZConv3d(3,3,AZConv3dConfig(
        modes=2,value_channels=4,kernel_size=3,contrastive=True,
        axial_preference=True,
    ))
    with torch.no_grad():
        model.mix.bias.zero_()
    image=torch.full((1,3,7,7,7),.35)
    output=model(image)
    # Weighted difference between neighbor and center is zero even at
    # patch boundaries; the unchanged shortcut therefore returns x.
    assert torch.allclose(output,image,atol=1e-6,rtol=0)


def test_contrastive_and_historical_operator_are_parameter_matched_ablation():
    torch.manual_seed(128)
    baseline=AZConv3d(3,3,AZConv3dConfig(
        modes=2,value_channels=4,kernel_size=3,
        axial_preference=True,
    ))
    contrast=AZConv3d(3,3,AZConv3dConfig(
        modes=2,value_channels=4,kernel_size=3,
        contrastive=True,axial_preference=True,
    ))
    contrast.load_state_dict(baseline.state_dict(),strict=True)
    x=torch.randn(1,3,6,6,6,requires_grad=True)
    old=baseline(x)
    new=contrast(x)
    assert old.shape==new.shape
    assert not torch.allclose(old,new,rtol=1e-5,atol=1e-5)
    assert count_trainable(contrast)==count_trainable(baseline)
    new.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert contrast.values.weight.grad is not None
    assert contrast.values.weight.grad.abs().sum()>0
