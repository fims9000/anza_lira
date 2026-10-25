import numpy as np, pytest
from scripts.research.bubble_chamber.s27_seam_crossing_injection import injected_long_track,seam_band,longest_run,measure

def test_injected_trace_labels_are_only_artificial_and_reproducible():
    src=np.random.default_rng(5).integers(75,230,(512,512),dtype=np.uint8)
    a,fg,scratch,coords=injected_long_track(src,42)
    b,fg2,scratch2,coords2=injected_long_track(src,42)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(fg,fg2)
    assert not np.any(fg&scratch)
    assert not np.array_equal(a,src)
    assert len(coords)>=400
    assert (seam_band(src.shape)&fg).any()
    assert (seam_band(src.shape)&scratch).any()
    p=measure(np.where(fg,1.0,0.0),fg,scratch,.5,coords)
    assert p['synthetic_roc_auc']==1.0
    assert p['overall_synthetic_tpr']==1.0
    assert p['overall_synthetic_scratch_fpr']==0.0
    assert p['physical_native_track_truth']=='UNKNOWN'

def test_longest_run_and_validation():
    assert longest_run(np.array([1,0,1,1,1,0],bool))==3
    assert longest_run(np.zeros(9,bool))==0
    with pytest.raises(ValueError):injected_long_track(np.zeros((512,512),dtype=np.float32),42)
