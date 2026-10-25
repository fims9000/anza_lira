import numpy as np
import cv2
import torch
import pytest
from bubble_chamber.s33_normalized_inference import centerline_evidence,patchwise_probabilities


class ConstantFakeModel(torch.nn.Module):
    def forward(self,x):
        return torch.full_like(x,.5)


def test_probability_map_complete_source_no_physics():
    source=np.full((512,512),180,np.uint8)
    model=ConstantFakeModel()
    p=patchwise_probabilities(model,source,mode='quantile',stride=64,batch_size=8)
    assert p.shape==(512,512)
    assert np.isfinite(p).all()
    assert np.allclose(p,float(torch.sigmoid(torch.tensor(.5))),atol=1e-5)
    q=patchwise_probabilities(model,source,mode='quantile',stride=32,batch_size=8)
    assert np.allclose(p,q,atol=1e-5)
    with pytest.raises(ValueError):patchwise_probabilities(model,source,mode='raw',stride=63)


def test_centerline_evidence_no_auto_class_and_source_bounds():
    prob=np.full((512,512),.5,np.float32)
    path=[[float(x),250.] for x in range(25,475,5)]
    r=centerline_evidence(prob,path)
    assert r['status']=='UNVERIFIED_MODEL_EVIDENCE'
    assert r['calibrated'] is False and r['median_center_minus_flank']==0.0
    bad=path.copy();bad[5]=[2000.,50.]
    with pytest.raises(ValueError):centerline_evidence(prob,bad)
    with pytest.raises(ValueError):centerline_evidence(prob[:20,:20],path)
