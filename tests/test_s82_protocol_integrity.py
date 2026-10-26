import json
from pathlib import Path
import numpy as np
import pytest
from scripts.research.bubble_chamber import s82_calibrate_per_line_then_test as s82


def test_seed_disjoint_and_frozen_conditions():
    prior={20267703,20267704,20267803,20267804,20267903,20267904,
           20268003,20268004,20268103,20268104,20268113,20268114}
    new=set(s82.TEST_SEEDS['005']+s82.TEST_SEEDS['006'])
    assert not new&prior and not new&set(s82.CAL_SEEDS.values())
    assert s82.TEST_SCENES==24 and s82.CAL_SCENES==20
    assert s82.TEST_CONDITIONS==('clean','blur_sigma_11','blur_sigma_16')


def test_inference_has_no_oracle_argument(monkeypatch):
    from bubble_chamber.s82_local_instance_support import ARMS
    # No candidate lines -> no graph proposal, but also no physical identity.
    baseline={'polylines':[],'chosen':[],'pair_count':0,'solver_success':True,
      'physics_auto_joins_authorized':False,'added_pixel_instances':0,
      'duplicate_suppressed':0,'scale_consensus_rejected':0,'original_polylines_retained':0}
    fake={'baseline':baseline,'s80_consensus_add6':dict(baseline)}
    monkeypatch.setattr(s82.s80,'infer_from_pixels',lambda *args:fake)
    image=np.full((192,192),205,np.uint8)
    thresholds={'s82_local_q25':.3,'s82_local_q50':.5}
    a=s82.infer_from_pixels(image,lambda x:np.zeros(len(x)),.8,'s39_original',thresholds)
    assert set(a)==set(ARMS)
    assert all(not a[x]['chosen'] and not a[x]['physics_auto_joins_authorized'] for x in ARMS)
    with pytest.raises(TypeError):s82.infer_from_pixels(image,lambda a:[],.8,'s39_original',thresholds,ground_truth=[])


def test_no_arbitrary_threshold_or_gt_reading():
    with pytest.raises(ValueError):s82.infer_from_pixels(np.ones((96,96),np.uint8),None,.8,'s39_original',{})
