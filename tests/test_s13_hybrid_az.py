import torch
import pytest
from bubble_chamber.s13_hybrid_az import GuidedUNet,instantiate_matched
from models.azconv import AZConv2d


def test_matched_backbone_and_gradients():
    torch.set_num_threads(2)
    models=instantiate_matched(123,widths=(4,4,4,4),rules=2)
    assert len(list(models['az_guided'].modules()))>len(list(models['unet_base'].modules()))
    assert any(isinstance(m,AZConv2d) for m in models['az_guided'].modules())
    assert not any(isinstance(m,AZConv2d) for m in models['conv_guided'].modules())
    for name,m in models.items():
        x=torch.randn(2,1,32,32)
        y=m(x)
        assert y.shape==(2,1,32,32) and torch.isfinite(y).all()
        y.square().mean().backward()
        # The original local_hyperbolic AZConv retains an unused isotropic
        # ablation scale; its grad is legitimately None in this mode.
        assert all(p.grad is not None or k.endswith('raw_sigma_iso')
                   for k,p in m.named_parameters() if p.requires_grad)
    assert 0<models['az_guided'].learned_guide_gate()<1


def test_original_az_parameters_unmodified_by_instantiation():
    with pytest.raises(ValueError):GuidedUNet(variant='unknown')
    with pytest.raises(ValueError):GuidedUNet(gate_init=1.0)
