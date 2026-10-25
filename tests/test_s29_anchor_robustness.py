import numpy as np
from scripts.research.bubble_chamber.s29_anchor_robustness_ablation import FIXED,recovery

def test_robustness_protocol_preregistered_no_human_truth():
    assert FIXED['jitter_y_px']==[0,3,6,10]
    assert FIXED['model_selection_using_new_trial_truth'] is False
    assert FIXED['physical_validation']=='None'
    assert FIXED['gap_condition']==[False,True]

def test_synthetic_recovery_coverage_range():
    t=np.array([[x,40.] for x in range(18,230)])
    guess={'path_points_original_xy':[[x,40.] for x in range(18,210)]}
    x=recovery([guess],t)
    assert x['top1']['prefix_coverage']>.9
    assert x['top1']['candidate_precision']>.9
    assert recovery([],t)['top1'] is None
