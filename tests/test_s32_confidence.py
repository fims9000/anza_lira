from __future__ import annotations
import numpy as np, pytest, torch
from bubble_chamber.s32_confidence import confidence_bce,balanced_weak_windows,patch_coverage
from scripts.research.bubble_chamber.s31_train_provisional_anza import split_bce


def test_weights_change_loss_and_unknown_has_no_effect():
    logits=torch.tensor([[[[-2.,2.,9.,-9.]]]],requires_grad=True)
    marks=torch.tensor([[[[1,0,255,255]]]])
    weights=torch.tensor([[[[.35,.03,0,0]]]])
    a=confidence_bce(logits,marks,weights)
    assert a is not None and bool(torch.isfinite(a))
    changed=weights.clone();changed[0,0,0,1]=.15
    b=confidence_bce(logits,marks,changed)
    assert float(a.detach())!=float(b.detach())
    # S31 deliberately ignores that field; show defect is fixed in S32.
    assert torch.equal(split_bce(logits,marks,weights),split_bce(logits,marks,changed))
    a.backward();assert logits.grad is not None
    assert torch.all(logits.grad[marks==255]==0)


def test_invalid_unknown_weight_and_labels_rejected():
    x=torch.ones(1,1,1,3);m=torch.tensor([[[[1,0,255]]]])
    with pytest.raises(ValueError,match='Unknown'):
        confidence_bce(x,m,torch.tensor([[[[.35,.03,.04]]]]))
    with pytest.raises(ValueError,match='Only machine'):
        confidence_bce(x,torch.tensor([[[[1,2,255]]]]),torch.zeros_like(x))
    with pytest.raises(ValueError,match='shape mismatch'):
        confidence_bce(x,m,torch.zeros(1,1,1,2))


def test_negative_windows_without_inventing_background():
    image=np.full((512,512),210,np.uint8)
    mask=np.full((512,512),255,np.uint8)
    mask[120:125,220:226]=1;mask[350:352,200:203]=0
    weights=np.zeros_like(image,np.float32)
    weights[mask==1]=.35;weights[mask==0]=.03
    rows=[(image,mask,weights,{'crop_id':'H50Z_030_01_005_test'})]
    a=balanced_weak_windows(rows,positives_per_crop=3,negatives_per_crop=4,seed=44)
    b=balanced_weak_windows(rows,positives_per_crop=3,negatives_per_crop=4,seed=44)
    assert len(a)==7
    assert all(np.array_equal(x[1],y[1]) for x,y in zip(a,b))
    cov=patch_coverage(a)
    assert cov['positive_centered']==3 and cov['negative_centered']==4
    assert cov['human_confirmed_labels']==0 and cov['unknown']>0
    for _,lab,w,_,_ in a:
        assert w[lab==255].sum()==0


def test_real_machine_manifest_source_lock_and_patch_counts():
    from pathlib import Path
    from scripts.research.bubble_chamber.s31_train_provisional_anza import load_weak_data
    manifest=Path('/mnt/data/s32/weak_training/S31_WEAK_MASKS_V2/weak_manifest.json')
    tiles=Path('/mnt/data/s32/photos_H50Z_derived')
    if not manifest.is_file() or not tiles.is_dir():pytest.skip('private S31 fixture unavailable in CI')
    groups,data=load_weak_data(manifest,tiles)
    train=balanced_weak_windows(groups['005'],positives_per_crop=2,negatives_per_crop=2)
    test=balanced_weak_windows(groups['006'],positives_per_crop=2,negatives_per_crop=2)
    assert train and test
    assert {r[3].split('_')[3] for r in train}=={'005'}
    assert {r[3].split('_')[3] for r in test}=={'006'}
    assert patch_coverage(test)['negative_centered']>0
