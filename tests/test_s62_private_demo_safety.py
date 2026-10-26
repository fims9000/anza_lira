import pytest
from scripts.research.bubble_chamber.s62_package_private_demo import assert_public_result,SCHEMA,SEEDS

def test_rejects_fake_physics_truth_and_incomplete_fresh_test():
    good={'schema':SCHEMA,'smoke_only_NOT_confirmatory':False,
          'native_H50Z_physical_track_GT_count':0,
          'source_polylines_never_provided_to_pixel_detectors':True,
          'directions':{'one':{'seeds':{str(x):{} for x in SEEDS}}}}
    assert_public_result(good)
    good['native_H50Z_physical_track_GT_count']=1
    with pytest.raises(ValueError):assert_public_result(good)
    good['native_H50Z_physical_track_GT_count']=0
    good['directions']['one']['seeds'].pop(str(SEEDS[-1]))
    with pytest.raises(ValueError):assert_public_result(good)
