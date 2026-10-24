import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from scripts.research.bubble_chamber.s17_mix_real_measured_shapes import sparse_for_measured_gray,build
from scripts.research.bubble_chamber.s17_make_book_prior_dataset import generate
from scripts.research.bubble_chamber.s17_eval_book_prior import _metric
from scripts.research.bubble_chamber.s14_train_sparse_anza import read_reviewed_manifest


def test_synthetic_measured_sparse_truth_not_background():
    im=np.full((128,128),220,np.uint8)
    truth=np.zeros_like(im);cv2.line(truth,(8,35),(119,58),1,2)
    image,labels=sparse_for_measured_gray(im,truth,np.random.default_rng(13))
    assert image.shape==labels.shape
    assert np.any(labels==1) and np.any(labels==0)
    assert np.any(labels==255)
    assert not np.any((labels==1)&(truth==0))
    assert not np.any((labels==0)&(truth!=0))
    # No assumption that the rest of the pixels are certified background.
    assert (labels==255).sum() > .95*labels.size


def test_dice_known_values_and_shape_rejection():
    y=np.array([[1,0],[0,1]])
    x=np.array([[.7,.8],[.3,.2]])
    d=_metric(x,y,threshold=.5)
    assert d['tp']==1 and d['fp']==1 and d['fn']==1 and d['dice']==.5
    with pytest.raises(ValueError,match='misaligned'):
        _metric(x,np.zeros((3,2)))


def test_full_mixed_manifest_preserves_film_holdout_label(tmp_path,monkeypatch):
    source=tmp_path/'book';out=tmp_path/'merged';generate(source,size=64,train=4,val=2,unlabeled=2,test=2)
    import scripts.research.bubble_chamber.s17_mix_real_measured_shapes as mod
    fake={'provenance_sha256':'GENERATED_FIXTURE','frames':[{'frame_id':f'070_{i:03d}'} for i in range(60)]}
    monkeypatch.setattr(mod,'audited_large_manifest',lambda _:fake)
    monkeypatch.setattr(mod,'frame_partition',lambda frames,seed:{
        'train':frames[:36],'val':frames[36:48],'test':frames[48:]})
    def fake_samples(manifest,ids,n,size,seed,split):
        for k in range(n):
            im=np.full((size,size),199,dtype='uint8')
            truth=np.zeros_like(im)
            cv2.line(truth,(5,5),(size-6,size-6),1,2)
            mask=np.full_like(im,255)
            mask[5:10,5:10]=1;mask[10:15,45:50]=0
            yield im,mask,truth,{'frame_id':ids[k], 'track_id':f't{k}', 'view':1}
    monkeypatch.setattr(mod,'_choose_measured',fake_samples)
    r=build(source,tmp_path,out,size=64,measured_train=6,measured_val=3,measured_test=3)
    assert r['book_train_fraction']==.4
    assert r['train_total']==10 and r['validation_total']==5 and r['synthetic_test_total']==5
    assert r['no_human_expert_dense_track_masks'] is True
    groups,meta=read_reviewed_manifest(out/'s17_book_mixed_sparse_manifest.json')
    assert len(groups['train'])==10 and len(groups['val'])==5
    assert all((m==255).any() for group in groups.values() for _,m,_ in group)
    assert set(row['frame_id'] for _,_,row in groups['train']).isdisjoint(
        row['frame_id'] for _,_,row in groups['val'])
