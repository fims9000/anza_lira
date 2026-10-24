from pathlib import Path
import numpy as np
import pytest
import torch
from bubble_chamber.s14_anza_frontend import AZInferenceConfig,az_mask_to_segments,load_original_az_checkpoint


def test_mask_to_segments_untrained_original_model_is_provisional():
    from models.azconv import AZConvConfig
    from models.segmentation import AZUNet
    torch.set_num_threads(2)
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
                 cfg=AZConvConfig(geometry_mode='local_hyperbolic'))
    img=np.full((64,64),200,dtype=np.uint8)
    img[10:54,30]=30
    res,mask,overlay=az_mask_to_segments(img,model)
    assert mask.shape==img.shape and overlay.shape==(64,64,3)
    assert res['pipeline']=='s14_original_azunet_sparse_review_v1'
    assert res['automatic_join_count']==0
    assert not res['supports_real_particle_identification']
    with pytest.raises(ValueError,match='too large'):
        az_mask_to_segments(img,model,AZInferenceConfig(max_input_pixels=1024))


def test_original_checkpoint_roundtrip(tmp_path):
    from models.azconv import AZConvConfig
    from models.segmentation import AZUNet
    from scripts.research.bubble_chamber.s14_train_sparse_anza import MODEL_SOURCE_BLOBS
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
                 cfg=AZConvConfig(geometry_mode='local_hyperbolic'))
    file=tmp_path/'test_best.pt'
    torch.save({'architecture':'azunet','model_weights':model.state_dict(),
                'source_git_blobs':MODEL_SOURCE_BLOBS,'rules':2,'architecture_widths':(4,4,4,4),
                'dataset_fingerprint':'a'*64,'epoch':1},file)
    read,meta=load_original_az_checkpoint(file)
    assert meta['checkpoint_epoch']==1
    assert set(read.state_dict())==set(model.state_dict())
