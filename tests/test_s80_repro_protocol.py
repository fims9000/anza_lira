import numpy as np
import pytest
from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80


def test_new_seed_holdout_does_not_reuse_old_protocols():
    old={20267703,20267704,20267713,20267714,
         20267803,20267804,20267813,20267814,
         20267903,20267904,20267913,20267914}
    seeds={k for x in s80.TEST_SEEDS.values() for k in x}
    assert seeds.isdisjoint(old)
    assert seeds.isdisjoint(set(s80.CAL_SEEDS.values()))
    assert (s80.CAL_SCENES,s80.TEST_SCENES)==(16,24)


def test_inference_API_never_accepts_oracle(monkeypatch):
    from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
    dummy={'polylines':[],'chosen':[],'pair_count':0,'solver_success':True}
    monkeypatch.setattr(s68,'infer_pixel_only',lambda *_:dummy)
    img=np.full((192,192),180,dtype=np.uint8)
    def scorer(rows):return np.ones(len(rows))
    x=s80.infer_from_pixels(img,scorer,.80,s68.ARMS[0])
    assert set(x)==set(s80.ARMS)
    assert all(not val['physics_auto_joins_authorized'] for val in x.values())
    with pytest.raises(TypeError):s80.infer_from_pixels(img,scorer,.8,s68.ARMS[0],truth=[])
    with pytest.raises(ValueError):s80.infer_from_pixels(img.astype(float),scorer,.8,s68.ARMS[0])


def test_frozen_lock_files_exist():
    assert s80.PROTOCOL.is_file() and s80.OPERATOR.is_file()
    assert s80.digest(s80.S72_CAL)==s80.LOCKED_S72_SHA256
