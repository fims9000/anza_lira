"""Ensure few-label training never invents background for a positive-only crop."""
import numpy as np
import torch
import pytest
from scripts.research.bubble_chamber import s14_train_sparse_anza as trainer
from bubble_chamber.s14_sparse_teacher import SparseTeacherProtocol


def test_original_sparse_train_with_disjoint_positive_only_and_negative_only_crops(tmp_path,monkeypatch):
    def toy_model(*args,**kwargs):
        return torch.nn.Conv2d(1,1,3,padding=1)
    monkeypatch.setattr(trainer,'build_model',toy_model)
    a=np.full((32,32),180,np.uint8);a[4:8,5:9]=50
    b=np.full((32,32),190,np.uint8);b[4:8,5:9]=30
    fg=np.full((32,32),255,np.uint8);fg[5:8,6:8]=1
    neg=np.full((32,32),255,np.uint8);neg[21:24,12:16]=0
    both=fg.copy();both[21:24,12:16]=0
    groups={'train':[(a,fg,{}),(b,neg,{})],'val':[(a,both,{})],'unlabeled':[]}
    report=trainer.fit_sparse_reviews(groups,{'source_manifest_sha256':'f'*64},out_dir=tmp_path,
         architecture='azunet',epochs=1,widths=(4,4,4,4),rules=2,
         teacher_cfg=SparseTeacherProtocol(consistency_weight=0.0))
    assert report['train_episodes']==1
    assert report['history'][0]['teacher_pseudo_pixels_used']==0
    ckpt=torch.load(tmp_path/'best.pt',map_location='cpu',weights_only=True)
    assert ckpt['source_data_private'] is True
    assert ckpt['history'][0]['known_validation_bce']>0


def test_single_crop_without_other_class_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_model',lambda *args,**kwargs:torch.nn.Conv2d(1,1,3,padding=1))
    im=np.full((32,32),150,np.uint8);mask=np.full((32,32),255,np.uint8);mask[5,5]=1
    with pytest.raises(ValueError,match='cannot form both-class batch'):
        trainer.fit_sparse_reviews({'train':[(im,mask,{})],'val':[(im,mask,{})],'unlabeled':[]},
            {'source_manifest_sha256':'f'*64},out_dir=tmp_path,epochs=1)
