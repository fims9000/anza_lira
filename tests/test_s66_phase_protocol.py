from pathlib import Path
import numpy as np
import pytest
from scripts.research.bubble_chamber.s66_test_image_only_endpoint_extensions import (
    CAL_SEEDS,TEST_SEEDS,CAL_SCENES,TEST_SCENES,image_only_inference_variants)

def test_frozen_s66_cal_and_test_seeds_do_not_reuse_prior():
    old={20266403,20266404,20266413,20266414,
         20266503,20266504,20266513,20266514}
    test=set(v for seq in TEST_SEEDS.values() for v in seq)
    assert not test&old and not test&set(CAL_SEEDS.values())
    assert CAL_SCENES==32 and TEST_SCENES==24


def test_pixel_inference_api_never_accepts_synthetic_label(monkeypatch):
    import scripts.research.bubble_chamber.s66_test_image_only_endpoint_extensions as mod
    from bubble_chamber.s66_image_endpoint_tracing import ARMS
    monkeypatch.setattr(mod,'predict_variants',lambda a:{'s39_original':[]})
    im=np.full((192,192),200,dtype=np.uint8)
    result=image_only_inference_variants(im,lambda rows:np.ones(len(rows)),.8)
    assert set(result)==set(ARMS)
    assert all(not z['chosen'] and not z['physics_auto_joins_authorized'] for z in result.values())
    with pytest.raises(TypeError):image_only_inference_variants(im,lambda a:[],.8,GT=[])
