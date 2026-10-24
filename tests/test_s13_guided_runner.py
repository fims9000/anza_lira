import json
from pathlib import Path
import torch
from scripts.research.bubble_chamber.s13_train_guided import run,ARMS
from scripts.research.bubble_chamber import s7_original_retrain as s7
from bubble_chamber.s13_hybrid_az import instantiate_matched


def test_four_registered_hybrid_arms():
    assert set(ARMS)=={'unet_base','conv_guided','az_guided','az_pure'}
    models=instantiate_matched(101,widths=(4,4,4,4),rules=2)
    b=sum(p.numel() for p in models['unet_base'].parameters())
    conv=sum(p.numel() for p in models['conv_guided'].parameters())
    az=sum(p.numel() for p in models['az_guided'].parameters())
    assert b<conv<az


def test_smoke_tiny_measured_geometry_dataset(tmp_path,monkeypatch):
    # Training algorithm and source lock remain genuine; only the generator
    # replaced with deterministic procedural data for quick CI and no private archive.
    from scripts.research.bubble_chamber import s13_train_guided as r
    cfg=s7.RunConfig(source='procedural',seed=991,data_seed=211,size=32,epochs=1,
                     batch_size=2,train_limit=4,val_limit=3,test_limit=4,
                     widths=(4,4,4,4),cpu_threads=2,az_residual=True,robust_train=False)
    result=r.run(cfg,None,tmp_path,arms=('unet_base','az_guided'))
    assert set(result['results'])=={'unet_base','az_guided'}
    assert result['source']['all_pixel_masks_simulated'] is True
    assert (tmp_path/'az_guided/best_final.pt').exists()
    assert (tmp_path/'unet_base/best_final.pt').exists()
