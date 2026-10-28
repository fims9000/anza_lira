"""Original AZUNet -> segmentation skeleton -> Graph-LIRA source-only contracts."""
from pathlib import Path
from hashlib import sha256
import numpy as np
import cv2
import pytest
import torch

from bubble_chamber.interop_anza_frontend import OriginalAZUNetDetector,verified_anza_to_graph
from bubble_chamber.s14_anza_frontend import AZInferenceConfig
from bubble_chamber.interop_api import process_image,ScoreBinding


class DarkStrokeMock(torch.nn.Module):
    """Deterministic input-to-logit mock, NOT AZUNet and NOT H50Z GT."""
    def forward(self,x):
        return 40*(x-.65)


def image():
    x=np.full((192,192),220,np.uint8)
    for y in (38,88,138):
        cv2.line(x,(24,y),(80,y),15,1)
        cv2.line(x,(95,y),(170,y),15,1)
    return x


def test_anza_mask_segments_from_model_flow_to_graph_without_real_physics_claim():
    im=image();original=im.copy()
    det=OriginalAZUNetDetector(DarkStrokeMock(),config=AZInferenceConfig(
        segmentation_threshold=.5,min_segment_length_px=15,points_per_fragment=14),max_lines=12)
    lines=det.detect(im)
    assert 1<=len(lines)<=12
    assert all(len(x)>=8 and not x.flags.writeable for x in lines)
    assert all(np.all((x>=0)&(x<=191)) for x in lines)
    assert np.array_equal(im,original)
    b=ScoreBinding('TOY_ONLY_UNTRAINED',det.detector_id,
                   lambda rows:np.full(len(rows),.8),cutoff=.75,
                   source_cal_id='TOY_NOT_CAL',strategy='none')
    res=process_image(im,det,b)
    assert len(res.lines)==len(lines) and res.detector_id==det.detector_id
    assert res.native_physics_gt_count==0
    assert not res.physical_auto_joins_authorized and not res.joins
    assert res.as_public_dict()['detected_lines']==len(lines)
    with pytest.raises(ValueError,match='Unverified/toy'):
        verified_anza_to_graph(im,det,receipt=None,model_file=Path('missing'),
                               source_cal_file=Path('missing'),score_pairs=None,
                               expected_training_domain='FAKE')


def test_anza_detector_identity_rejects_unverified_non_toy_and_parameter_drift():
    model=DarkStrokeMock()
    with pytest.raises(ValueError,match='TOY_ONLY'):
        OriginalAZUNetDetector(model,detector_id='incorrect',toy_only=True)
    with pytest.raises(ValueError,match='digest'):
        OriginalAZUNetDetector(model,detector_id='original-azunet-invalid',toy_only=False)
    with pytest.raises(ValueError,match='line budget'):
        OriginalAZUNetDetector(model,max_lines=33)
    with pytest.raises(FileNotFoundError):
        OriginalAZUNetDetector.from_private_checkpoint(Path('/definitely-no-h50z-checkpoint.pt'))


def test_actual_original_azunet_architecture_available_without_trained_weights():
    from models.azconv import AZConvConfig,AZConv2d
    from models.segmentation import AZUNet,BaselineUNet
    torch.set_num_threads(2)
    cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_anisotropy=True,use_fuzzy=True)
    anza=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),cfg=cfg)
    baseline=BaselineUNet(in_channels=1,out_channels=1,widths=(4,4,4,4))
    assert any(isinstance(m,AZConv2d) for m in anza.modules())
    assert not any(isinstance(m,AZConv2d) for m in baseline.modules())
    with torch.no_grad():
        assert anza(torch.zeros((1,1,32,32))).shape==(1,1,32,32)
        assert baseline(torch.zeros((1,1,32,32))).shape==(1,1,32,32)
    assert not hasattr(anza,'frozen_h50z_real_gt')


def test_real_checkpoint_source_loader_is_strict_and_does_not_synthesize_training():
    from scripts.research.bubble_chamber.s14_train_sparse_anza import verify_model_source
    assert len(verify_model_source())==2
    detector=OriginalAZUNetDetector(DarkStrokeMock(),max_lines=10)
    assert detector.research_only
    assert detector.checkpoint_sha256 is None


def test_verified_original_private_checkpoint_stable_identity_and_never_native_claims(tmp_path):
    from models.azconv import AZConvConfig
    from models.segmentation import AZUNet
    from scripts.research.bubble_chamber.s14_train_sparse_anza import MODEL_SOURCE_BLOBS
    torch.set_num_threads(2)
    model=AZUNet(in_channels=1,out_channels=1,num_rules=2,widths=(4,4,4,4),
                 cfg=AZConvConfig(geometry_mode='local_hyperbolic',use_fuzzy=True,
                                  use_anisotropy=True))
    f=tmp_path/'PRIVATE_TEST_ONLY_AZUNET_UNTRAINED.pt'
    torch.save({'architecture':'azunet','model_weights':model.state_dict(),
                'source_git_blobs':MODEL_SOURCE_BLOBS,'rules':2,
                'architecture_widths':(4,4,4,4),'dataset_fingerprint':'a'*64,
                'epoch':0},f)
    det,m=OriginalAZUNetDetector.from_private_checkpoint(f,max_lines=6)
    assert not det.toy_only and det.checkpoint_sha256==sha256(f.read_bytes()).hexdigest()
    assert m['checkpoint_sha256']==det.checkpoint_sha256
    assert m['native_physics_gt_count']==0 and not m['automatic_real_joins_authorized']
    changed_config=AZInferenceConfig(segmentation_threshold=.65)
    alt,_=OriginalAZUNetDetector.from_private_checkpoint(f,config=changed_config,max_lines=6)
    assert alt.detector_id != det.detector_id
    assert len(det.detect(image()))<=6
    with torch.no_grad():
        first=next(det.model.parameters())
        first.add_(.01)
    with pytest.raises(ValueError,match='weights changed'):
        det.detect(image())
    # Merely loading an untrained checkpoint is NOT evidence of performance.
    assert m['checkpoint_epoch']==0
