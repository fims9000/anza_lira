"""V55 training and full-section inference contracts on synthetic source arrays."""
import numpy as np
import pytest
import torch

from scripts.tver_v55_fullslice_segmenter import (
    _patch, _draw_batch, _profile, predict_complete,train_model,
    exact_top_fraction_mask,
)

SCALE={
    "TRAIN_only_1pct":0.,"TRAIN_only_99pct":1.,
    "normalized_channel_mean":0.,"normalized_channel_std":1.,
}


def sample_volume():
    seismic=np.zeros((5,96,96),np.float32)
    for i in range(5):
        seismic[i,:,:]=float(i)/4.
    label=np.zeros_like(seismic,bool)
    label[:,25:75,47]=True
    return seismic,label


def test_2d_is_center_only_but_2p5d_sees_only_observed_neighbors():
    x,y=sample_volume()
    two,label=_patch(x,y,2,50,47,"2d",SCALE)
    hybrid,_=_patch(x,y,2,50,47,"2p5d",SCALE)
    assert np.allclose(two[0],two[1]) and np.allclose(two[1],two[2])
    assert hybrid[0].mean()<hybrid[1].mean()<hybrid[2].mean()
    assert np.array_equal(label,y[2,18:82,15:79])


def test_identical_training_center_schedule_and_finite_one_step():
    x,y=sample_volume()
    batch2,targets=_draw_batch(
        x,y,SCALE,np.random.default_rng(7),"2d",2,{})
    batch25,targets25=_draw_batch(
        x,y,SCALE,np.random.default_rng(7),"2p5d",2,{})
    assert np.array_equal(targets,targets25)
    assert np.allclose(batch2[:,0],batch2[:,1])
    net,receipt=train_model(x,y,SCALE,"2d",steps=1,batch_size=2,seed=7)
    assert receipt["steps"]==1
    assert np.isfinite(receipt["mean_last100_training_loss"])
    assert net.training is False


def test_multichannel_overlap_blending_is_exact_for_pointwise_model():
    x,_=sample_volume()
    source=np.stack([x[1],x[2],x[3]],axis=0)
    net=torch.nn.Conv2d(3,1,1,bias=False)
    with torch.no_grad():
        net.weight.zero_()
        net.weight[0,0,0,0]=1.
        net.weight[0,1,0,0]=2.
        net.weight[0,2,0,0]=3.
    pred=predict_complete(source,net,SCALE,"2p5d",
                          tile=32,stride=23,batch=4)
    expected=float(torch.sigmoid(torch.tensor(.25+2*.5+3*.75)))
    assert pred.shape==source.shape[1:]
    assert np.allclose(pred,expected,atol=1e-6)
    assert len(_profile(pred,np.zeros_like(pred,bool)))==4


def test_cannot_train_or_infer_using_invalid_context():
    x,y=sample_volume()
    with pytest.raises(ValueError,match="Invalid"):
        _patch(x,y,0,50,50,"2p5d",SCALE)
    net=torch.nn.Conv2d(3,1,1)
    with pytest.raises(ValueError,match="three aligned"):
        predict_complete(x,net,SCALE,"2d",tile=32,stride=16)


def test_tied_probability_plateau_still_obeys_exact_pixel_budget():
    p=np.full((40,50),.6734,np.float32)
    p[:5,:]=.8
    chosen,cutoff=exact_top_fraction_mask(p,.02)
    assert int(chosen.sum())==40
    assert chosen[:1,:40].all()
    assert not chosen[1:].any()
    assert cutoff==float(np.float32(.8))
    assert np.array_equal(chosen,exact_top_fraction_mask(p,.02)[0])
    with pytest.raises(ValueError,match="exact positive"):
        exact_top_fraction_mask(p,1.2)
