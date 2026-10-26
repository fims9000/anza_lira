"""Immutable CAL before fresh TEST, no hidden evaluation-label selection."""
from pathlib import Path
import json
import pytest
from scripts.research.bubble_chamber.s65_calibrate_endpoint_abstention import (
    CAL_SEEDS,TEST_SEEDS,CAL_SCENES,TEST_SCENES,_lock_calibration,SCHEMA)

def test_s65_split_seeds_different_from_s64_and_each_other():
    old={20266403,20266404,20266413,20266414}
    new=set(sum((list(v) for v in TEST_SEEDS.values()),[]))
    assert not old.intersection(new)
    assert not new.intersection(CAL_SEEDS.values())
    assert (CAL_SCENES,TEST_SCENES)==(32,24)

def test_calibration_loader_rejects_test_or_fake_schema(tmp_path):
    path=tmp_path/'cal.json'
    path.write_text(json.dumps({'schema':SCHEMA,'phase':'test'}))
    with pytest.raises(ValueError):_lock_calibration(path)
    path.write_text(json.dumps({'schema':'not_s65','phase':'cal'}))
    with pytest.raises(ValueError):_lock_calibration(path)
    path.write_text(json.dumps({'schema':SCHEMA,'phase':'cal'}))
    obj,sha=_lock_calibration(path)
    assert obj['phase']=='cal' and len(sha)==64
