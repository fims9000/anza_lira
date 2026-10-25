import numpy as np
import torch
import pytest
from pathlib import Path
from scripts.research.bubble_chamber import s14_train_sparse_anza as train
from scripts.research.bubble_chamber.s20_pretrain_unlabeled_real import Config,run
from bubble_chamber.s20_transfer import transfer_unlabeled_encoder
from tests.test_s20_unlabeled import fake_source_photos
from bubble_chamber.s14_sparse_teacher import SparseTeacherProtocol


def get_real_unlabeled_checkpoint(tmp_path,architecture):
    fake_source_photos(tmp_path/'tiles')
    cfg=Config(seed=20261025,epochs=1,batch=2,patch=32,train_per_tile=2,
               val_per_tile=1,widths=(4,4,4,4),rules=2)
    result=run(tmp_path/'tiles',tmp_path/'pretext',cfg,models=(architecture,))
    return tmp_path/'pretext'/architecture/'best.pt'


def partially_confirmed_source_masks():
    img=np.full((32,32),180,np.uint8)
    img[5:11,9:14]=80
    fg=np.full((32,32),255,np.uint8);fg[5:8,10:12]=1
    bg=np.full((32,32),255,np.uint8);bg[23:27,21:25]=0
    both=fg.copy();both[23:27,21:25]=0
    return {'train':[(img,fg,{}),(img,bg,{})],
            'val':[(img,both,{})], 'unlabeled':[]}


@pytest.mark.parametrize('architecture',['unet','azunet'])
def test_finetune_reuses_unlabeled_encoder_without_claiming_real_labels(tmp_path,architecture):
    ck=get_real_unlabeled_checkpoint(tmp_path,architecture)
    groups=partially_confirmed_source_masks()
    summary={'source_manifest_sha256':'a'*64}
    report=train.fit_sparse_reviews(groups,summary,out_dir=tmp_path/'finetuned',architecture=architecture,
        epochs=1,widths=(4,4,4,4),rules=2,init_unlabeled_encoder=ck,
        teacher_cfg=SparseTeacherProtocol(consistency_weight=0.))
    info=report['unlabeled_encoder_pretraining']
    assert info is not None and info['architecture']==architecture
    assert info['reconstruction_head_imported'] is False
    assert report['validation_metric'].startswith('BCE on CERTIFIED pixels')
    checkpoint=torch.load(tmp_path/'finetuned'/'best.pt',weights_only=True)
    assert checkpoint['unlabeled_encoder_pretraining']['checkpoint_sha256']==info['checkpoint_sha256']
    assert checkpoint['history'][0]['teacher_pseudo_pixels_used']==0
    assert checkpoint['source_data_private'] is True
    assert info['requires_human_sparse_labels'] is True


def test_pretraining_cannot_bypass_necessary_expert_classes(tmp_path):
    ck=get_real_unlabeled_checkpoint(tmp_path,'azunet')
    im=np.zeros((32,32),dtype='uint8')
    fg=np.full((32,32),255,dtype='uint8');fg[10,10]=1
    with pytest.raises(ValueError,match='both-class'):
        train.fit_sparse_reviews({'train':[(im,fg,{})],'val':[(im,fg,{})],'unlabeled':[]},
                                 {'source_manifest_sha256':'0'*64},out_dir=tmp_path/'bad',epochs=1,
                                 architecture='azunet',widths=(4,4,4,4),rules=2,init_unlabeled_encoder=ck)


def test_s19_end_to_end_review_to_encoder_finetune_on_toy_photo(tmp_path):
    """Fake reviewers and fake photo pixels are fixture-only, never real human labels."""
    from tests.test_s19_real_review import fixture
    from bubble_chamber.s19_real_review import load_package,empty_review
    from scripts.research.bubble_chamber.s19_run_small_label_pipeline import run_pipeline
    fake_source_photos(tmp_path/'source_toy')
    cfg=Config(seed=20261025,epochs=1,batch=2,patch=32,train_per_tile=2,
               val_per_tile=1,widths=(4,4,4,4),rules=2)
    run(tmp_path/'source_toy',tmp_path/'pret',cfg,models=('azunet','unet'))
    (tmp_path/'review_fixture').mkdir()
    proposals,images=fixture(tmp_path/'review_fixture')
    package=load_package(proposals,images)
    review=empty_review(package)
    for tile,entry in zip(package['tiles'],review['entries']):
        entry['strokes']=[
            {'id':f'manual:{tile["tile_id"]}:0','label':'track_centerline',
             'radius_px':1,'points_xy':[[5,14],[15,14]]},
            {'id':f'manual:{tile["tile_id"]}:1','label':'certified_artifact',
             'radius_px':1,'points_xy':[[5,47],[15,47]]},
        ]
    review_path=tmp_path/'test_not_expert.json'
    review_path.write_text(__import__('json').dumps(review))
    result=run_pipeline(proposals=proposals,images=images,reviews=review_path,
        out_dir=tmp_path/'finetune',epochs=1,widths=(4,4,4,4),rules=2,patch_size=32,
        init_azunet=tmp_path/'pret/azunet/best.pt',
        init_unet=tmp_path/'pret/unet/best.pt')
    assert result['status']=='REAL_REVIEW_TRAINING_COMPLETED_PROVISIONAL_ONLY'  # contract name, only fake test data
    assert result['real_automatic_track_merges_allowed'] is False
    assert result['graph']['human_confirmed_pairs']==0
    for name in ('azunet','unet'):
        m=result['trained_models'][name]
        assert m['unlabeled_pretraining']['reconstruction_head_imported'] is False
        assert m['real_full_image_pixel_dice']=='NOT_MEASURED'
        assert (tmp_path/'finetune'/name/'best.pt').is_file()
