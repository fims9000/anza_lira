"""S18 synthetic-only controlled physics-prior ablation contracts."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.research.bubble_chamber.s18_book_ablation import (
    Config,Sample,background_clicks,choose_threshold,make_dataset,
    paired_frame_bootstrap,run_arm,sparse_supervised_loss,stress_images
)


def test_book_substitution_is_same_budget_disjoint_frames():
    root=Path(__file__).resolve().parents[1]/'inputs/1mH2Data/S1'
    if not root.is_dir():pytest.skip('private measured geometries not part of git checkout')
    cfg=Config(examples_train=12,shared_measured=8,examples_val=5,examples_test=6,book_test=3)
    d,r=make_dataset(root,cfg,style_crops=None)
    a,b=d['A_measured_only'],d['B_measured_plus_book']
    assert len(a)==len(b)==12
    assert all(np.array_equal(a[i].image,b[i].image) and np.array_equal(a[i].sparse,b[i].sparse)
               for i in range(8))
    assert len(set(s.group for s in a) & set(s.group for s in d['test_measured'])) == 0
    assert len(set(s.group for s in a) & set(s.group for s in d['validation_measured'])) == 0
    assert all(s.source_kind.startswith('book') for s in b[8:])
    assert r['source_only_one_film'] is True
    assert r['real_film_images_used_for_training'] is False


def test_background_clicks_only_synthetic_unknown_negative():
    sparse=np.full((10,10),255,dtype=np.uint8)
    sparse[2,2]=1;sparse[7,7]=0
    truth=np.zeros_like(sparse);truth[2,2]=1;truth[1:5,4]=1
    out=background_clicks(sparse,truth,seed=43,count=20)
    assert np.count_nonzero(out==1)==1
    assert np.count_nonzero(out==0)==21
    assert np.all(truth[out==0]==0)
    assert np.array_equal(out,background_clicks(sparse,truth,seed=43,count=20))
    with pytest.raises(ValueError):background_clicks(sparse,truth,seed=1,count=-1)
    with pytest.raises(ValueError):background_clicks(sparse,truth[:-1],seed=1,count=20)


def test_unknown_pixels_have_no_gradient():
    logits=torch.zeros(1,1,3,3,requires_grad=True)
    y=torch.full((1,1,3,3),255,dtype=torch.int64)
    y[0,0,0,0]=1;y[0,0,2,2]=0
    loss=sparse_supervised_loss(logits,y);loss.backward()
    assert torch.isfinite(loss).all()
    assert logits.grad[0,0,0,0]!=0
    assert logits.grad[0,0,2,2]!=0
    assert torch.all(logits.grad[0,0,1,:]==0)
    with pytest.raises(ValueError):sparse_supervised_loss(logits,y*0+255)


def test_paired_source_frame_bootstrap_identical_zero():
    y=np.zeros((5,12,12),np.uint8)
    y[:,3:10,5]=1
    p=y.astype(np.float32)
    groups=['f1','f1','f2','f3','f3']
    res=paired_frame_bootstrap(p,p,y,groups,.5,.5,seed=101,draws=80)
    assert res['source_frames']==3
    assert res['delta_dice']==0
    assert res['ci95_frame_bootstrap']==[0.,0.]


def test_stress_does_not_change_original_and_is_seeded():
    cfg=Config()
    x=torch.rand(3,1,64,64,generator=torch.Generator().manual_seed(7))
    saved=x.clone()
    a=stress_images(x,cfg,seed=101)
    b=stress_images(x,cfg,seed=101)
    assert torch.equal(a,b)
    assert torch.equal(x,saved)
    assert a.min()>=0 and a.max()<=1
    assert not torch.equal(a,x)


def test_threshold_not_informed_by_test():
    p=np.array([[[.1,.3],[.8,.9]]],np.float32)
    y=np.array([[[0,0],[1,1]]],np.uint8)
    assert 0.15 <= choose_threshold(p,y) <= .9


def test_tiny_original_model_checkpoint_and_dense_sim_evaluation(tmp_path):
    # Verify original unchanged models can be built and trained end-to-end.
    from scripts.research.bubble_chamber.s17_make_book_prior_dataset import make_example
    def examples(role,n):
        out=[]
        for i in range(n):
            im,truth,sparse,_,_=make_example(64,np.random.default_rng(500+i+(0 if role=='train' else 100)), 'liquid_hydrogen')
            out.append(Sample(im,background_clicks(sparse,truth,seed=i,count=12),truth,role+str(i),'synthetic',role+str(i)))
        return out
    c=Config(epochs=1,batch_size=2,size=64,examples_train=4,shared_measured=2,examples_val=2,examples_test=2,book_test=2)
    report=run_arm('toy_book','azunet',examples('train',4),examples('val',2),examples('test',2),examples('book',2),cfg=c,out=tmp_path)
    assert (tmp_path/'best.pt').exists()
    assert (tmp_path/'heldout_probabilities.npz').exists()
    assert report['source_only_one_film'] is True  # training source flag on this research pipeline
    assert report['real_bmp_pixel_metrics'] is None


def test_generic_style_control_never_uses_heldout_pixels():
    root=Path(__file__).resolve().parents[1]/'inputs/1mH2Data/S1'
    if not root.is_dir():pytest.skip('private measured geometries not part of git checkout')
    cfg=Config(examples_train=12,shared_measured=8,examples_val=4,examples_test=4,book_test=2)
    d,_=make_dataset(root,cfg,style_crops=None)
    b=d['B_measured_plus_book'];generic=d['D_measured_book_synthetic_style']
    assert len(b)==len(generic)==12
    for i in range(8):
        assert np.array_equal(b[i].image,generic[i].image)
    for j in range(8,12):
        assert np.array_equal(b[j].truth,generic[j].truth)
        assert np.array_equal(b[j].sparse,generic[j].sparse)
        assert generic[j].source_kind=='book_hadroide_train_synthetic_histogram'
        assert not np.array_equal(b[j].image,generic[j].image)


def test_generic_stress_control_holds_labels_constant_without_real_style():
    root=Path(__file__).resolve().parents[1]/'inputs/1mH2Data/S1'
    if not root.is_dir():pytest.skip('private measured geometries not part of git checkout')
    cfg=Config(examples_train=12,shared_measured=8,examples_val=4,examples_test=4,book_test=2)
    d,summary=make_dataset(root,cfg,style_crops=None)
    assert summary['real_reference_style_unlabeled_sha256']==[]
    e=d['E_book_generic_robust_augmentation'];b=d['B_measured_plus_book']
    assert len(e)==len(b)
    for j in range(8,12):
        assert e[j].source_kind=='book_hadroide_generic_noise_contrast_no_real_refs'
        assert np.array_equal(e[j].sparse,b[j].sparse)
        assert np.array_equal(e[j].truth,b[j].truth)
        assert not np.array_equal(e[j].image,b[j].image)
