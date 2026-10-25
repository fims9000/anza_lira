import numpy as np
import cv2
import pytest
from bubble_chamber.s33_photometric_domain import image_input,photometric_augment,input_domain_audit


def test_affine_exposure_invariance_without_clipping():
    rng=np.random.default_rng(17)
    im=rng.integers(45,185,size=(64,64),dtype=np.uint8)
    altered=np.rint(.80*im.astype(float)+26).astype(np.uint8)
    raw=np.mean(np.abs(image_input(im,'raw')-image_input(altered,'raw')))
    quantile=np.mean(np.abs(image_input(im,'quantile')-image_input(altered,'quantile')))
    local=np.mean(np.abs(image_input(im,'local')-image_input(altered,'local')))
    assert raw>.025
    assert quantile<.015
    assert local<.015


def test_source_intensity_invariant_but_not_ground_truth():
    a=np.tile(np.linspace(100,200,64,dtype=np.uint8),(64,1))
    out=image_input(a,'local')
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    assert image_input(a,'raw').dtype==np.float32
    for mode in ('quantile','local'):
        assert out.shape==image_input(a,mode).shape
    with pytest.raises(ValueError):image_input(a,'unknown')
    with pytest.raises(ValueError):image_input(a.astype(float))
    with pytest.raises(ValueError):image_input(np.zeros((12,12),np.uint8))


def test_explicit_augmentation_deterministic_and_bounded():
    source=np.full((64,64),155,np.uint8)
    a=photometric_augment(source,seed=123)
    assert np.array_equal(a,photometric_augment(source,seed=123))
    assert a.dtype==np.uint8 and a.shape==source.shape
    assert not np.array_equal(a,source)


def test_domain_audit_cannot_use_labels():
    im=np.full((64,64),120,np.uint8)
    result=input_domain_audit({'005':[im], '006':[im+20]})
    assert result['006']['mean_gray']-result['005']['mean_gray']==20
    assert 'physical_accuracy' not in str(result)
    with pytest.raises(ValueError):input_domain_audit({})


def test_original_weak_photo_loader_does_not_change_source_or_label_semantics(tmp_path):
    from scripts.research.bubble_chamber.s33_train_photometric import load_source_weak_masks
    from pathlib import Path
    import json
    # Self-contained micro-fixture has 39 source crops but no human GT.
    source=tmp_path/'tiles';source.mkdir();masks=tmp_path/'masks';masks.mkdir()
    entries=[]
    from hashlib import sha256
    for i in range(39):
        p='005' if i<21 else '006';name=f'H50Z_030_01_{p}_fake{i:02d}'
        img=np.full((512,512),180,np.uint8);mask=np.full((512,512),255,np.uint8)
        weights=np.zeros((512,512),np.float16);mask[100:110,200]=1;weights[100:110,200]=.35
        im=source/(name+'.png');m=masks/(name+'_weak.png');w=masks/(name+'_weights.npy')
        cv2.imwrite(str(im),img);cv2.imwrite(str(m),mask);np.save(w,weights)
        entries.append({'crop_id':name,'photo_id':p,'image_sha256':sha256(im.read_bytes()).hexdigest(),
            'mask':'masks/'+m.name,'mask_sha256':sha256(m.read_bytes()).hexdigest(),
            'weight':'masks/'+w.name,'weight_sha256':sha256(w.read_bytes()).hexdigest()})
    mfile=tmp_path/'weak.json'
    mfile.write_text(json.dumps({'schema':'s31_weak_training_masks_source_locked_v1',
        'human_certified_track_pixels':0,'physics_auto_join_allowed':False,'entries':entries}))
    groups,receipt=load_source_weak_masks(mfile,source)
    assert len(groups['005'])==21 and len(groups['006'])==18
    assert receipt['physically_certified_track_pixels']==0
    # Unknown (255) does not become a negative.
    assert np.all(groups['006'][0][2][groups['006'][0][1]==255]==0)
    entries[0]['mask_sha256']='0'*64
    mfile.write_text(json.dumps({'schema':'s31_weak_training_masks_source_locked_v1',
        'human_certified_track_pixels':0,'physics_auto_join_allowed':False,'entries':entries}))
    with pytest.raises(ValueError,match='SHA-256'):load_source_weak_masks(mfile,source)
