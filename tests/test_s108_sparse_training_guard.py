"""Synthetic S108 regression: no unlabeled negatives or test-metric claims."""
import cv2
import numpy as np
import pytest
import torch

from bubble_chamber.s108_sparse_unet import (
    S108UNet, image_channels, sparse_masked_bce, predict_score_preview,
    research_pilot_train,
)


def test_s108_loss_uses_only_explicit_strokes():
    logits=torch.zeros((1,1,32,32),requires_grad=True)
    labels=torch.zeros((1,32,32),dtype=torch.long)
    labels[0,7,7]=1;labels[0,8,8]=2;labels[0,9,9]=3
    val=sparse_masked_bce(logits,labels)
    val.backward()
    grad=logits.grad[0,0]
    assert grad[7,7]!=0 and grad[8,8]!=0
    assert grad[9,9]==0 and grad[0,0]==0
    labels.zero_()
    with pytest.raises(ValueError,match='No explicitly'):
        sparse_masked_bce(logits,labels)


def test_s108_smoke_forward_backward_without_real_gt(tmp_path):
    torch.manual_seed(7)
    model=S108UNet(base=8)
    gray=np.full((128,128),201,np.uint8)
    cv2.line(gray,(12,24),(111,100),135,2)
    inputs=image_channels(gray)
    assert inputs.shape==(2,128,128)
    x=torch.from_numpy(inputs[None])
    label=torch.zeros((1,128,128),dtype=torch.long)
    label[:,30:50,20:30]=1
    label[:,80:100,20:30]=2
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    out=model(x)
    loss=sparse_masked_bce(out,label)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    weights=tmp_path/'weights.pt'
    torch.save({'state_dict':model.state_dict(),'base':8},weights)
    restored=S108UNet(base=8)
    restored.load_state_dict(torch.load(weights,weights_only=True)['state_dict'])
    rank=predict_score_preview(restored,gray,patch=64,overlap=16)
    assert rank.shape==gray.shape and np.isfinite(rank).all()


def test_s108_rejects_same_run_as_independent_test(tmp_path):
    with pytest.raises(ValueError,match='one run'):
        research_pilot_train(tmp_path/'missing_pack',
                             tmp_path/'missing_annotation.json',
                             tmp_path/'weights', steps=1)
