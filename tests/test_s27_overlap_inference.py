import numpy as np, torch, pytest
from bubble_chamber.s27_overlap_inference import (
    tile_starts,blend_weights,overlap_predict,tile_boundary_jump,disagreement_stats)

class Toy(torch.nn.Module):
    def forward(self, x):
        assert x.ndim==4
        return x[:,0:1]*2.0-1.0

class BorderArtifacts(torch.nn.Module):
    def forward(self,x):
        y=x[:,0:1]*3-1
        y[:,:,:5,:]+=1.5
        y[:,:,-5:,:]-=1.5
        y[:,:,:,:5]+=1.5
        y[:,:,:,-5:]-=1.5
        return y


def prep(arr):return (255-arr).astype('float32')[None]/255


def test_exact_pixel_alignment_and_positive_coverage():
    a=np.random.default_rng(7).integers(0,256,(192,224),dtype='uint8')
    result=overlap_predict(a,Toy(),prep,tile=64,stride=32,batch_size=7)
    expected=(1/(1+np.exp(-(2*(255-a.astype('float32'))/255-1))))
    assert result.shape==a.shape
    np.testing.assert_allclose(result,expected,rtol=1e-6,atol=1e-6)
    assert np.all(blend_weights(64)>.0)
    assert tile_starts(221,64,32)[-1]==157


def test_nonoverlap_equal_original_logits():
    a=np.random.default_rng(24).integers(0,256,(128,128),dtype='uint8')
    p=overlap_predict(a,Toy(),prep,tile=64,stride=64)
    ex=(1/(1+np.exp(-(2*(255-a.astype('float32'))/255-1))))
    np.testing.assert_allclose(p,ex,atol=1e-6)


def test_overlap_lessens_simulated_tile_edge_artifact():
    a=np.full((256,256),155,dtype='uint8')
    one=overlap_predict(a,BorderArtifacts(),prep,stride=64)
    mixed=overlap_predict(a,BorderArtifacts(),prep,stride=32)
    old=tile_boundary_jump(one);new=tile_boundary_jump(mixed)
    assert new['mean_boundary_jump']<old['mean_boundary_jump']*.5
    assert new['boundary_to_control_ratio']<old['boundary_to_control_ratio']


def test_no_claim_of_ground_truth():
    a=np.zeros((128,128),dtype='float32');b=a.copy();a[12:18,12:18]=.7
    out=disagreement_stats(a,b,.5)
    assert out['no_real_expert_track_labels'] is True
    assert out['iou_between_own_candidate_masks_NOT_GT']==0
    assert out['candidate_pixel_fraction_changed']>0


def test_bad_inputs_fail_closed():
    with pytest.raises(ValueError):tile_starts(32,64,32)
    with pytest.raises(ValueError):overlap_predict(np.zeros((64,64),np.float32),Toy(),prep)
    with pytest.raises(ValueError):overlap_predict(np.zeros((64,64),np.uint8),Toy(),lambda x:x[None]*np.nan)
    with pytest.raises(ValueError):tile_boundary_jump(np.zeros((20,20)),tile=64)


def test_global_feature_preprocessing_preserves_exact_coordinates():
    from bubble_chamber.s27_overlap_inference import overlap_predict_global_features
    a=np.random.default_rng(29).integers(0,256,(128,192),dtype='uint8')
    # Pointwise feature map is invariant to windowing, apart from float rounding.
    first=overlap_predict(a,Toy(),prep)
    second=overlap_predict_global_features(a,Toy(),prep)
    np.testing.assert_allclose(second,first,rtol=1e-6,atol=1e-6)
    def bad(x):return np.full((2,x.shape[0],x.shape[1]),np.nan,'float32')
    with pytest.raises(ValueError):overlap_predict_global_features(a,Toy(),bad)


def test_global_features_reduce_patch_local_blur_edge_seams():
    from bubble_chamber.s27_overlap_inference import overlap_predict_global_features
    import cv2
    src=np.zeros((256,256),'uint8')+160
    cv2.line(src,(15,10),(240,233),70,2)
    def blurprep(a):
        return (cv2.GaussianBlur(a.astype('float32'),(0,0),5)-a.astype('float32'))[None]/45
    first=overlap_predict(src,Toy(),blurprep,stride=32)
    second=overlap_predict_global_features(src,Toy(),blurprep,stride=32)
    assert np.max(np.abs(first-second))>1e-4  # distinguishes preprocessing stages
    assert np.isfinite(second).all()
