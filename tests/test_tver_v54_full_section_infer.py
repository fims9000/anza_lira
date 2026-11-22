"""Tiled full-section inference is pixel-exact for translation-equivariant toy models."""
import numpy as np
import pytest
import torch

from scripts.tver_v54_full_section_infer import tile_starts, tiled_probability

SCALE={
    "TRAIN_only_1pct":0.,"TRAIN_only_99pct":1.,
    "normalized_channel_mean":0.,"normalized_channel_std":1.,
}


def test_tiling_no_holes_and_no_seams_when_image_shape_not_multiple_of_stride():
    rng=np.random.default_rng(57)
    pixels=rng.uniform(0,1,size=(117,139)).astype(np.float32)
    model=torch.nn.Conv2d(3,1,1,bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0,0,0,0]=1.
    model.eval()
    result,receipt=tiled_probability(pixels,model,SCALE,
                                     tile_size=32,stride=21,batch_size=4)
    reference=torch.sigmoid(torch.from_numpy(pixels)).numpy()
    assert np.allclose(result,reference,rtol=0,atol=1e-6)
    assert result.shape==pixels.shape
    assert receipt["n_tiles"]==len(tile_starts(117,32,21))*len(tile_starts(139,32,21))
    assert receipt["uses_expert_truth_as_input"] is False


def test_tiled_inference_freezes_training_scale_not_per_section_scale():
    image=np.ones((56,57),dtype=np.float32)*.75
    scale=dict(SCALE, TRAIN_only_1pct=.2, TRAIN_only_99pct=.8,
               normalized_channel_mean=.1,normalized_channel_std=.5)
    model=torch.nn.Conv2d(3,1,1,bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0,0,0,0]=1.
    out,_=tiled_probability(image,model,scale,tile_size=32,stride=16,batch_size=2)
    expected=float(torch.sigmoid(torch.tensor(((.75-.2)/(.8-.2)-.1)/.5)))
    assert np.allclose(out,expected,atol=1e-6)


def test_invalid_tiling_or_nonfinite_fails_closed():
    model=torch.nn.Conv2d(3,1,1)
    with pytest.raises(ValueError,match="tiling"):
        tile_starts(20,32,16)
    image=np.ones((64,64),np.float32)
    image[0,0]=np.nan
    with pytest.raises(ValueError,match="complete"):
        tiled_probability(image,model,SCALE,tile_size=32,stride=16)
    image[0,0]=0
    with pytest.raises(ValueError,match="normalization"):
        tiled_probability(image,model,dict(SCALE,TRAIN_only_99pct=0),
                          tile_size=32,stride=16)
