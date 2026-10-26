"""S42 source-provenance and reproducible ablation safety tests."""
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.research.bubble_chamber import s42_train_pure_teacher_ablation as a


def test_training_batch_is_deterministic_with_unknown_ignored(monkeypatch):
    monkeypatch.setattr(a,'EPOCHS',1)
    monkeypatch.setattr(a,'STEPS',2)
    monkeypatch.setattr(a,'BATCH',2)
    image=np.full((512,512),180,dtype=np.uint8)
    image[20:80,20:100]=60
    mask=np.full((512,512),255,dtype=np.uint8)
    mask[20:80,20:100]=1
    mask[50:130,50:160]=0
    mask[20:36,20:36]=1
    x=a._draw_batches([(image,mask)],'005')
    y=a._draw_batches([(image,mask)],'005')
    assert len(x)==len(y)==2
    assert all(batch[0].shape==(2,1,64,64) and batch[1].shape==(2,1,64,64) for batch in x)
    assert all(torch.equal(left[0],right[0]) and torch.equal(left[1],right[1]) for left,right in zip(x,y))
    assert all(torch.any(b[1]==255) for b in x)


def test_resume_mismatched_checkpoint_rejected(tmp_path):
    # Refuse to call a model trained on another source photo 'photo disjoint'.
    f=tmp_path/f'S42_unet_train_photo_005_seed_{a.SEED}.pt'
    torch.save({'architecture':'unet','source_photo':'006','photo_teacher_type':'geometry-only, no AZUNet',
                'physics_native_gt':False,'epochs':a.EPOCHS,'batch':a.BATCH,
                'steps':a.STEPS,'seed':a.SEED},f)
    with pytest.raises(ValueError,match='incompatible checkpoint'):
        a.train_original_model('unet',[],tmp_path,'005')


def test_frozen_protocol_keeps_disjoint_test_photo_and_synthetic_scenes():
    assert a.SEED==20261026
    assert a.N_SCENES==24
    assert a.TEST_SEEDS=={'005':20261129,'006':20261130}
    assert a.STEPS==32 and a.EPOCHS==3 and a.BATCH==8
