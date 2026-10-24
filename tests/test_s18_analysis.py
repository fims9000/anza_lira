import numpy as np
import pytest
from scripts.research.bubble_chamber.s18_analyze_results import per_frame_counts, hierarchical_frame_seed_bootstrap

def test_frame_bootstrap_identical_prediction_parity():
    a={0:{'f1':np.array([10,2,3]),'f2':np.array([7,4,2])},1:{'f1':np.array([20,4,3]),'f2':np.array([4,2,1])}}
    result=hierarchical_frame_seed_bootstrap(a,a,draws=90)
    assert result['pooled_delta_dice']==0
    assert result['paired_ci95_seed_and_frame']==[0.,0.]
    assert result['test_unique_source_frames']==2
    with pytest.raises(ValueError):hierarchical_frame_seed_bootstrap(a,{0:a[0]},draws=90)

def test_confusion_grouping_is_frame_not_pixel():
    p=np.zeros((3,8,8),float);y=np.zeros((3,8,8),np.uint8);y[:,1,1]=1;p[0,1,1]=1
    groups=['f1','f1','f2']
    r=per_frame_counts(p,y,groups,.5)
    assert len(r)==2
    assert np.array_equal(r['f1'],[1,0,1]);assert np.array_equal(r['f2'],[0,0,1])
