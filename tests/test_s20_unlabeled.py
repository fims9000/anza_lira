import hashlib
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from bubble_chamber.s20_unlabeled import (discover_real_tiles,UnlabeledPhotoPatches, mask_blocks,
    render_masked_sample,masked_metrics,inpaint_baseline)
from bubble_chamber.s20_transfer import transfer_unlabeled_encoder
from scripts.research.bubble_chamber.s20_pretrain_unlabeled_real import Config,run,verify_original_sources,build_model


def fake_source_photos(folder):
    folder.mkdir(parents=True,exist_ok=True)
    names=['H50Z_030_01_005_start0_tile.png','H50Z_030_01_005_start1_tile.png',
           'H50Z_030_01_006_start0_tile.png']
    for i,n in enumerate(names):
        yy,xx=np.mgrid[:80,:80]
        a=np.clip(145+(xx+yy)/4+np.sin(xx/4+i)*30+i*15,0,255).astype('uint8')
        assert cv2.imwrite(str(folder/n),a)
    return names


def test_real_grouping_and_source_identity(tmp_path):
    names=fake_source_photos(tmp_path)
    photos,receipt=discover_real_tiles(tmp_path)
    assert receipt['train_crops']==2 and receipt['validation_crops']==1
    assert receipt['real_human_annotation_count']==0
    assert receipt['train_photo_id']!=receipt['validation_photo_id']
    assert all(len(x.sha256)==64 for x in photos)
    # No silent recognition of another photo or invalid color data.
    (tmp_path/'H50Z_other.png').write_bytes(b'not a photo')
    with pytest.raises(ValueError,match='Unrecognized'):
        discover_real_tiles(tmp_path)


def test_generator_leaves_hidden_pixels_unknown_not_expert_mask(tmp_path):
    fake_source_photos(tmp_path)
    tiles,info=discover_real_tiles(tmp_path)
    a=UnlabeledPhotoPatches(tiles[:1],patch=32,per_tile=3,seed=13)
    b=UnlabeledPhotoPatches(tiles[:1],patch=32,per_tile=3,seed=13)
    assert torch.equal(a.x,b.x) and torch.equal(a.hole,b.hole)
    assert len(a)==3
    x,y,h=a[0]
    assert h.dtype==torch.bool
    assert torch.all(x[h]==0.5)
    assert torch.allclose(x[~h],y[~h])
    assert 0.01<float(h.float().mean())<0.7
    assert not torch.allclose(x,y)
    assert a.provenance[0]['synthetic_missing_pixels']>0
    cv2.imwrite(str(tiles[0].path),np.zeros((80,80),dtype=np.uint8))
    with pytest.raises(ValueError,match='changed'):
        UnlabeledPhotoPatches(tiles[:1],patch=32,per_tile=1)


def test_masked_metrics_and_classical_reference():
    im=np.full((32,32),170,dtype=np.uint8);im[7:25,14:17]=50
    holes=mask_blocks(32,np.random.default_rng(12),fraction=.22,block=4)
    x,y,holes=render_masked_sample(im,holes)
    mm=masked_metrics(y,y,holes)
    assert mm['masked_mse']==0 and mm['masked_psnr_db']>=100
    baseline=inpaint_baseline(x,holes)
    assert baseline.shape==x.shape
    out=masked_metrics(baseline,y,holes)
    assert out['pixels_scored']>0 and np.isfinite(out['masked_mae'])
    with pytest.raises(ValueError):masked_metrics(x,y,np.zeros((3,3)))


def test_source_locked_and_finetune_head_is_fresh(tmp_path):
    verify_original_sources()
    fake_source_photos(tmp_path/'tiles')
    cfg=Config(seed=120,epochs=1,batch=2,patch=32,train_per_tile=2,val_per_tile=1,
               widths=(4,4,4,4),rules=2)
    result=run(tmp_path/'tiles',tmp_path/'out',cfg,models=('azunet','unet'))
    assert set(result['methods'])=={'azunet','unet'}
    assert result['real_track_labels_used'] is False
    assert result['independent_physical_run_test']=='NOT_AVAILABLE'
    assert result['methods']['azunet']['validation_reconstruction']['pixels_scored']>0
    for name in ('azunet','unet'):
        ck=tmp_path/'out'/name/'best.pt'
        assert ck.exists()
        model=build_model(name,cfg)
        saved_head={k:v.clone() for k,v in model.state_dict().items() if k.startswith('head.')}
        info=transfer_unlabeled_encoder(model,ck,architecture=name,widths=cfg.widths,rules=cfg.rules)
        assert info['reconstruction_head_imported'] is False
        assert info['requires_human_sparse_labels'] is True
        state=model.state_dict()
        assert all(torch.equal(state[k],v) for k,v in saved_head.items())
        assert any(not torch.equal(state[k],torch.load(ck,weights_only=True)['weights'][k]) for k in saved_head) or not saved_head
        with pytest.raises(ValueError,match='mismatch'):
            transfer_unlabeled_encoder(model,ck,architecture=name,widths=(8,8,8,8),rules=2)
    with pytest.raises(ValueError,match='Untrusted'):
        bad=tmp_path/'bad.pt';torch.save({'schema':'fake'},bad)
        transfer_unlabeled_encoder(build_model('azunet',cfg),bad,architecture='azunet',widths=cfg.widths,rules=cfg.rules)
