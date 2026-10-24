import numpy as np
import pytest
import torch

from bubble_chamber.s11_axial_topology import (
    StructuralWeights, soft_skeleton, cldice_loss, structural_training_loss,
    binary_cldice, per_frame_confusion, count_metrics, paired_frame_bootstrap
)


def truth_mask(size=32):
    y = torch.zeros(1,1,size,size)
    y[:,:,9,4:28] = 1
    y[:,:,21,4:28] = 1
    return y


def test_differentiable_cldice_with_nonempty_grad():
    t=truth_mask()
    logits=torch.zeros_like(t,requires_grad=True)
    obj=cldice_loss(logits,t,iterations=3)
    assert torch.isfinite(obj) and 0 <= obj <= 1
    obj.backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum())>0


def test_soft_skeleton_input_checks():
    with pytest.raises(ValueError):soft_skeleton(torch.ones(1,2,32,32))
    with pytest.raises(ValueError):soft_skeleton(torch.ones(1,1,32,32),iterations=0)
    with pytest.raises(ValueError):cldice_loss(torch.zeros(1,1,32,32),torch.zeros(1,2,32,32))
    with pytest.raises(ValueError):StructuralWeights(cldice=-1).validate()


def test_true_cldice_penalizes_false_bridge():
    y=truth_mask().numpy()[0,0].astype(bool)
    p=y.copy()
    p[9:22,18]=True
    broken=y.copy()
    broken[9,12:21]=False
    assert binary_cldice(y,y)==1.0
    assert 0 < binary_cldice(p,y)<1
    assert 0 < binary_cldice(broken,y)<1
    assert binary_cldice(np.zeros((32,32),bool),np.zeros((32,32),bool))==1
    assert binary_cldice(np.zeros((32,32),bool),y)==0


def test_original_az_axis_alignment_differentiable():
    from models.segmentation import AZUNet
    from models.azconv import AZConvConfig
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
       cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_input_residual=True,residual_init=.5))
    inp=torch.randn(1,1,32,32)
    target=truth_mask()
    logits=model(inp)
    base=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    weights=StructuralWeights(cldice=.2,axis=.05,skeleton_iterations=2,axis_iterations=2)
    full,parts=structural_training_loss(logits,target,model,base,weights)
    assert torch.isfinite(full)
    assert float(parts['axis'])>=0
    full.backward()
    # Direction learning should affect original AZ geometry parameters (not just decoder).
    grads=[p.grad for n,p in model.named_parameters() if 'geometry_conv' in n]
    assert grads and any(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in grads)


def test_plain_unet_fair_cldice_without_axis():
    from models.segmentation import BaselineUNet
    model=BaselineUNet(in_channels=1,out_channels=1,widths=(4,4,4,4))
    inp=torch.randn(1,1,32,32)
    y=truth_mask()
    logits=model(inp)
    base=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
    full,_=structural_training_loss(logits,y,model,base,StructuralWeights(cldice=.2,axis=0))
    assert torch.isfinite(full)
    with pytest.raises(ValueError):structural_training_loss(logits,y,model,base,StructuralWeights(cldice=.2,axis=.1))


def test_bootstrap_frame_identity_not_pixel():
    y=np.zeros((10,32,32),bool);y[:,9,5:25]=True
    p=np.zeros_like(y)
    p[:5]=y[:5]
    frame=[f'f_{i:03d}' for i in range(10)]
    perfect=per_frame_confusion(y,y,frame)
    partial=per_frame_confusion(p,y,frame)
    assert count_metrics(perfect)['dice']==1
    a=paired_frame_bootstrap(partial,perfect,draws=80)
    assert a['candidate_minus_baseline_dice']>0
    assert a['ci95'][0]>=0
    assert a['frames']==10
    with pytest.raises(ValueError):paired_frame_bootstrap(perfect,dict(list(partial.items())[:9]))
