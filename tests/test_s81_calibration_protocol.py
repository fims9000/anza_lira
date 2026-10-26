import numpy as np
import pytest
from scripts.research.bubble_chamber import s81_calibrate_quality_gate as s81


def test_new_seeds_and_original_model_locked():
    old={20267903,20267904,20267913,20267914,20268003,20268004,20268013,20268014}
    assert not {x for v in s81.TEST_SEEDS.values() for x in v}&old
    assert not {x for v in s81.TEST_SEEDS.values() for x in v}&set(s81.CAL_SEEDS.values())
    assert (s81.CAL_SCENES,s81.TEST_SCENES)==(16,24)
    assert s81.digest(s81.SOURCE_S80_CAL)


def test_image_inference_has_no_gt_parameter(monkeypatch):
    from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80
    source={'polylines':[],'chosen':[],'pair_count':0,'solver_success':True,
            'physics_auto_joins_authorized':False,'added_pixel_instances':0,
            'duplicate_suppressed':0,'scale_consensus_rejected':0,'original_polylines_retained':0}
    monkeypatch.setattr(s80,'infer_from_pixels',lambda *args:{'baseline':source,'s80_consensus_add6':source})
    g=np.full((192,192),180,dtype=np.uint8)
    choices=s81.infer_from_pixels(g,lambda rows:np.ones(len(rows)),.8,'original',
        {'s81_gate_q20':0.,'s81_gate_q40':1.})
    assert set(choices)==set(s81.ARMS)
    assert choices['baseline']['quality_gate_triggered']==0
    with pytest.raises(TypeError):s81.infer_from_pixels(g,None,.8,'original',{},source_nodes=[])
    with pytest.raises(ValueError):s81.infer_from_pixels(g.astype(float),None,.8,'original',{})
