import pytest
import torch
from bubble_chamber.s14_sparse_teacher import build_teacher_targets,sparse_teacher_loss,SparseTeacherProtocol


def sample():
    human=torch.full((1,1,8,8),255,dtype=torch.uint8)
    human[0,0,1,1]=1
    human[0,0,6,6]=0
    t1=torch.full((1,1,8,8),.5)
    t2=t1.clone()
    t1[0,0,0,0]=t2[0,0,0,0]=.995
    t1[0,0,7,7]=t2[0,0,7,7]=.005
    return human,t1,t2


def test_confidence_pseudolabels_only_unknown():
    labels,t1,t2=sample()
    t1[0,0,1,1]=t2[0,0,1,1]=.001 # teacher contradicts expert
    t1[0,0,6,6]=t2[0,0,6,6]=.999
    pseudo,mask,s=build_teacher_targets(labels,t1,t2)
    assert pseudo[0,0,1,1]==255 and pseudo[0,0,6,6]==255
    assert pseudo[0,0,0,0]==1 and pseudo[0,0,7,7]==0
    assert (mask&(labels!=255)).sum()==0
    assert s['expert_pixels_overwritten']==0


def test_disagreement_blocks_pseudo_supervision():
    labels,t1,t2=sample()
    t2[0,0,0,0]=.2
    p,m,s=build_teacher_targets(labels,t1,t2)
    assert not m.any() and (p==255).all()
    assert s['used_positive_count']==0


def test_unknown_pixels_ignored_and_no_teacher_gradients():
    labels,t1,t2=sample()
    # no pseudo labels, a change to an UNKNOWN pixel cannot change human loss
    low=torch.zeros((1,1,8,8),requires_grad=True)
    other=low.detach().clone();other[0,0,3,3]=20;other.requires_grad_()
    cfg=SparseTeacherProtocol(consistency_weight=0)
    a,stats_a=sparse_teacher_loss(low,labels,t1,t2,cfg)
    b,stats_b=sparse_teacher_loss(other,labels,t1,t2,cfg)
    assert torch.allclose(a,b)
    a.backward()
    assert float(low.grad[0,0,3,3])==0
    assert stats_a['human_positive_pixels']==1


def test_teacher_consistency_has_student_grad_and_is_bounded():
    labels,t1,t2=sample()
    logits=torch.zeros((1,1,8,8),requires_grad=True)
    loss,s=sparse_teacher_loss(logits,labels,t1,t2)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0,0,0,0]!=0 and logits.grad[0,0,7,7]!=0
    assert s['used_positive_count']==1 and s['used_negative_count']==1
    assert s['pseudo_weight']<=0.5


def test_no_unreviewed_negatives_or_forged_background():
    labels,t1,t2=sample()
    labels[0,0,6,6]=255
    with pytest.raises(ValueError,match='both expert-confirmed'):
        sparse_teacher_loss(torch.zeros((1,1,8,8)),labels,t1,t2)
    with pytest.raises(ValueError):
        build_teacher_targets(labels,t1,t2,split='test')
    bad=labels.clone();bad[0,0,0,1]=2
    with pytest.raises(ValueError):build_teacher_targets(bad,t1,t2)


def test_full_weighted_loss_on_original_AZUNet_cpu():
    from models.azconv import AZConvConfig
    from models.segmentation import AZUNet
    torch.set_num_threads(2)
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
                 cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_anisotropy=True,use_fuzzy=True))
    x=torch.randn(1,1,32,32)
    target=torch.full((1,1,32,32),255,dtype=torch.uint8)
    target[0,0,8:18,8]=1;target[0,0,8:18,20]=0
    with torch.no_grad():
        teacher=torch.sigmoid(model(x)).detach()
    logits=model(x)
    loss,s=sparse_teacher_loss(logits,target,teacher,teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert s['human_positive_pixels']==10 and s['human_negative_pixels']==10
