"""S9 regression checks for source-locked ANZA contrast training."""
import numpy as np
import torch
import pytest
from scripts.research.bubble_chamber.s7_original_retrain import (
    RunConfig, build_models, robust_augment, robust_validation_threshold, measured_examples, verify_sources
)

def test_exact_original_sources():
    assert len(verify_sources()) == 2

def test_train_only_augmentation_is_deterministic():
    cfg=RunConfig(robust_train=True,robust_probability=1.0)
    x=torch.full((3,1,32,32),0.45)
    ids=torch.tensor([0,1,2])
    assert torch.equal(robust_augment(x,ids,cfg,4),robust_augment(x,ids,cfg,4))
    assert not torch.equal(robust_augment(x,ids,cfg,4),robust_augment(x,ids,cfg,5))

def test_robust_validation_threshold_shape_guard():
    a=np.ones((2,4,4),np.float32); y=a.copy()
    assert .05 <= robust_validation_threshold(a,a/3,y) <= .95
    with pytest.raises(ValueError,match="must agree"):
        robust_validation_threshold(a,a[:1],y)

def test_original_azconv_residual_is_used():
    cfg=RunConfig(size=32,widths=(4,4,4,4),num_rules=2,az_residual=True)
    model=build_models(cfg)['original_azunet']
    layers=[m for m in model.modules() if type(m).__name__=='AZConv2d']
    assert len(layers)==3 and all(m.use_input_residual for m in layers)
    x=torch.randn(2,1,32,32,requires_grad=True)
    model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()

def test_balanced_frame_selection(monkeypatch,tmp_path):
    from bubble_chamber import large_archive,large_relation_study,synthetic_from_clicks
    frames=[f'070_{n:03d}' for n in range(40)]
    monkeypatch.setattr(large_archive,'audited_large_manifest',lambda path: {
        'frames':[{'frame_id':f} for f in frames],
        'provenance_sha256':'fixture','summary':{'distinct_film_ids':['070']}})
    monkeypatch.setattr(large_relation_study,'frame_partition',lambda ids,seed:
        {'train':ids[:20],'val':ids[20:30],'test':ids[30:40]})
    def many(manifest,ids,cfg,repeats):
        for f in ids:
            for i in range(20):
                yield np.full((64,64),i,np.uint8),np.zeros((64,64),np.uint8),{'source_frame':f}
    monkeypatch.setattr(synthetic_from_clicks,'frame_render_dataset',many)
    cfg=RunConfig(source='visualmetro',size=64,train_limit=20,val_limit=8,test_limit=8)
    ds,_=measured_examples(tmp_path,cfg)
    assert {k:len(set(t[2] for t in v)) for k,v in ds.items()}=={'train':20,'val':8,'test':8}
    assert not set(t[2] for t in ds['train']) & set(t[2] for t in ds['test'])
