import numpy as np
import pytest
from scripts.research.bubble_chamber import s83_calibrate_non_saturating as s83


def test_frozen_NEW_seed_splits_not_reused_from_s82():
    old={20268201,20268211,20268203,20268204,20268213,20268214}
    source=set(s83.CAL_SEEDS.values())
    heldout=set(s83.TEST_SEEDS['005']+s83.TEST_SEEDS['006'])
    assert not old & source and not old & heldout and not source & heldout
    assert s83.CAL_SCENES==20 and s83.TEST_SCENES==24
    assert s83.TEST_CONDITIONS==('clean','blur_sigma_11','blur_sigma_16')


def test_inference_requires_pixels_only_and_precise_thresholds(monkeypatch):
    baseline={'polylines':[],'chosen':[],'pair_count':0,'solver_success':True,
      'physics_auto_joins_authorized':False,'added_pixel_instances':0,
      'duplicate_suppressed':0,'scale_consensus_rejected':0,'original_polylines_retained':0}
    monkeypatch.setattr(s83.s80,'infer_from_pixels',lambda *a:{'baseline':baseline,'s80_consensus_add6':dict(baseline)})
    gray=np.full((192,192),205,np.uint8)
    gates={'s83_local_q40':.2,'s83_local_q70':.3}
    allarms=s83.infer_from_pixels(gray,lambda a:np.ones(len(a)),.7,'s39_original',gates)
    assert set(allarms)==set(s83.ARMS)
    assert all(not x['chosen'] and not x['physics_auto_joins_authorized'] for x in allarms.values())
    with pytest.raises(TypeError):s83.infer_from_pixels(gray,None,.7,'s39_original',gates,true_nodes=[])
    with pytest.raises(ValueError):s83.infer_from_pixels(gray,None,.7,'s39_original',{})
