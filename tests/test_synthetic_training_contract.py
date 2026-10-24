import torch
import numpy as np
import importlib.util
from pathlib import Path


def module():
    import sys
    p=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/run_synthetic_anza_pretrain.py'
    s=importlib.util.spec_from_file_location('run_synthetic_anza_pretrain',p)
    mod=importlib.util.module_from_spec(s);sys.modules[s.name]=mod;s.loader.exec_module(mod)
    return mod


def test_bce_plus_dice_gradient():
    m=module()
    logits=torch.zeros((1,1,16,16),requires_grad=True)
    y=torch.zeros_like(logits)
    y[:,:,5:9,7:9]=1
    loss=m.segmentation_loss(logits,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_val_threshold_cannot_use_test_truth():
    m=module()
    val_p=np.array([.7,.35,.2,.8],dtype=float)
    val_y=np.array([1,0,0,1],dtype=np.uint8)
    chosen=m.threshold_from_validation(val_p,val_y)
    assert chosen>.3 and chosen<=.7
    metrics=m.pixel_metrics(val_p,val_y,chosen)
    assert metrics['dice']>.7
