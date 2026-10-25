"""No H50Z raw bytes, no real photo truth, never force simulated labels to gold."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from dataclasses import replace

from bubble_chamber.s35_photometric_consistency import (
    S35Protocol, paired_exposure_views, consistency_loss, view_disagreement)
from scripts.research.bubble_chamber.s35_train_consistency import make_matched_views
from scripts.research.bubble_chamber.s35_eval_consistency import _model_load


def source_marks():
    x=torch.zeros((1,1,6,6),dtype=torch.float32)
    y=torch.full((1,1,6,6),255,dtype=torch.long)
    weight=torch.zeros_like(x)
    y[0,0,0,0]=1;weight[0,0,0,0]=.35
    y[0,0,5,5]=0;weight[0,0,5,5]=.03
    return x,y,weight


def test_predeclared_protocol_no_physical_truth():
    protocol=S35Protocol();assert protocol.check() is True
    assert not protocol.test_real_native_track_truth
    assert not protocol.auto_physics_join_authorized
    with pytest.raises(ValueError,match='preregistered'):
        replace(protocol,epochs=50).check()
    with pytest.raises(ValueError,match='physical'):
        replace(protocol,test_real_native_track_truth=True).check()
    with pytest.raises(ValueError,match='preregistered'):
        replace(protocol,test_reverse_seeds=protocol.test_forward_seeds).check()
    with pytest.raises(ValueError,match='preregistered'):
        replace(protocol,lambda_consistency=.5).check()
    with pytest.raises(ValueError,match='preregistered'):
        replace(protocol,positives_per_crop=20).check()


@pytest.mark.parametrize('mode',['quantile','local'])
def test_consistency_views_same_geometry_and_reproducible(mode):
    g=np.random.default_rng(20)
    gray=g.integers(35,240,size=(64,64),dtype=np.uint8)
    cv1,cv2=paired_exposure_views(gray,mode,seed=44)
    other1,other2=paired_exposure_views(gray,mode,seed=44)
    assert cv1.shape==(64,64) and cv2.shape==(64,64)
    assert np.array_equal(cv1,other1) and np.array_equal(cv2,other2)
    assert np.isfinite(cv1).all() and np.isfinite(cv2).all()
    assert 0<=cv1.min() and cv1.max()<=1
    with pytest.raises(ValueError,match='Unregistered'):
        paired_exposure_views(gray,'raw',seed=44)


def test_no_supervised_gradients_on_unknown_pixels_without_consistency():
    x,y,w=source_marks();a=x.clone().requires_grad_();b=x.clone().requires_grad_()
    loss,diag=consistency_loss(a,b,y,w,include_consistency=False)
    loss.backward()
    known=y!=255
    assert diag['consistency_unknown_pixels']==0
    assert torch.all(a.grad[~known]==0) and torch.all(b.grad[~known]==0)
    assert torch.any(a.grad[known]!=0) and torch.any(b.grad[known]!=0)


def test_consistency_uses_unknown_only_without_manufacturing_ground_truth():
    x,y,w=source_marks()
    a=torch.ones_like(x,requires_grad=True)
    b=torch.zeros_like(x,requires_grad=True)
    loss,diag=consistency_loss(a,b,y,w,include_consistency=True,alpha=.1)
    assert diag['consistency_unknown_pixels']==34
    assert diag['consistency']>0
    assert diag['supervised_unknown_count']==0
    loss.backward()
    assert torch.all(a.grad[y==255]==0) # detached teacher on unlabeled sites
    assert (b.grad[y==255].abs()>0).any()
    assert torch.isfinite(loss)


def test_empty_machine_marks_are_not_implicitly_negative():
    x=torch.zeros((1,1,32,32));y=torch.full((1,1,32,32),255,dtype=torch.long)
    with pytest.raises(ValueError,match='machine labeled'):
        consistency_loss(x,x,y,torch.zeros_like(x))
    y[0,0,1,1]=1
    with pytest.raises(ValueError,match='zero weight'):
        consistency_loss(x,x,y,torch.ones_like(x))


def test_augmented_views_preserve_unknown_mark_locations():
    rng=np.random.default_rng(44)
    gray=rng.integers(0,255,size=(64,64),dtype=np.uint8)
    mask=np.full((64,64),255,dtype=np.uint8);mask[2:4,8:12]=1;mask[10:15,15:20]=0
    conf=np.zeros((64,64),np.float32);conf[mask==1]=.35;conf[mask==0]=.03
    rows=[(gray,mask,conf,'dummy_crop',1)]
    a,b,t,w=make_matched_views(rows,[0],seed=20261025,epoch=0,mode='quantile')
    assert a.shape==b.shape==(1,1,64,64)
    assert int((t==255).sum())==64*64-8-25
    assert torch.all(w[t==255]==0)
    assert not torch.equal(a,b)


def test_view_disagreement_rejects_non_probs():
    p=np.full((32,32),.5);q=p+.02
    d=view_disagreement(p,q)
    assert d['mean_absolute_probability_delta']>0
    assert d['real_track_accuracy']=='NOT_MEASURED'
    with pytest.raises(ValueError,match='Expected probability'):
        view_disagreement(p,q+2)
    with pytest.raises(ValueError,match='Noncomparable'):
        view_disagreement(p,q[:16])


def test_refuse_nonprovenance_checkpoint(tmp_path):
    import torch
    filename=tmp_path/'azunet_matched_two_view_train005_seed20261025.pt'
    torch.save({'schema':'real_expert_GOLD','architecture':'azunet','mode':'quantile'},filename)
    with pytest.raises(ValueError,match='provenance'):
        _model_load(tmp_path,'azunet','matched_two_view','005',20261025)
