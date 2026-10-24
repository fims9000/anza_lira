import torch
import pytest
from bubble_chamber.s12_orientation_field import reference_orientation,az_orientation_field_loss


def test_reference_horizontal_and_vertical_direction():
    z=torch.zeros((1,1,64,64))
    z[:,:,32,10:54]=1
    c,s,w=reference_orientation(z)
    core=w[0,0,32,15:49]>0.05
    assert bool(core.any())
    assert float(c[0,0,32,15:49][core].mean())>0.3
    v=torch.zeros_like(z);v[:,:,10:54,32]=1
    vc,vs,vw=reference_orientation(v)
    core2=vw[0,0,15:49,32]>0.05
    assert bool(core2.any())
    assert float(vc[0,0,15:49,32][core2].mean())<-.3


def test_original_az_training_receives_direction_gradients():
    from models.segmentation import AZUNet
    from models.azconv import AZConvConfig
    torch.manual_seed(2026)
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
      cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_input_residual=True,residual_init=.5))
    y=torch.zeros((2,1,32,32));y[:,:,9,4:28]=1;y[:,:,22,4:28]=1
    x=torch.rand_like(y)
    with pytest.raises(ValueError):az_orientation_field_loss(model,y)
    _=model(x)
    value,diag=az_orientation_field_loss(model,y)
    assert diag['valid_stages']>=1 and float(value)>0
    model.zero_grad(set_to_none=True)
    value.backward()
    norm=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
             if p.grad is not None and 'geometry_conv' in n)
    assert norm>1e-4, norm


def test_zero_truth_skips_invalid_tangent_instead_of_inventing_ground_truth():
    from models.segmentation import AZUNet
    from models.azconv import AZConvConfig
    m=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
      cfg=AZConvConfig(geometry_mode='local_hyperbolic'))
    _=m(torch.randn((1,1,32,32)))
    loss,d=az_orientation_field_loss(m,torch.zeros((1,1,32,32)))
    assert d['valid_stages']==0 and float(loss)==0
