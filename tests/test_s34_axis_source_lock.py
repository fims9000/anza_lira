import torch
import numpy as np
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model
from scripts.research.bubble_chamber.s34_train_axis import FIXED_AXIS_WEIGHT,FIXED_ITERATIONS


def test_axis_loss_routes_gradients_to_original_azconv_geometry():
    torch.manual_seed(41);torch.set_num_threads(2)
    m=build_model('azunet',widths=(4,6,8,12),rules=2)
    x=torch.rand((2,1,64,64),requires_grad=False)
    y=torch.zeros_like(x)
    y[:,:,:,31:33]=1
    out=m(x)
    loss=m.axis_alignment_loss(y,torch.ones_like(y),num_iters=FIXED_ITERATIONS)
    assert loss.isfinite() and loss.item()>0
    (out.mean()+FIXED_AXIS_WEIGHT*loss).backward()
    assert m.enc1.block[0].geometry_conv.weight.grad is not None
    assert torch.isfinite(m.enc1.block[0].geometry_conv.weight.grad).all()


def test_axis_supervision_must_not_claim_human_truth():
    import inspect
    from scripts.research.bubble_chamber.s34_train_axis import fit
    code=inspect.getsource(fit)
    assert 'axial_target_is_machine_derived' in code
    assert 'pixel_labels_are_human_certified' in code
    assert 'test_injection_seeds_reserved' in code
