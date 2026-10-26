import inspect
from pathlib import Path
import pytest
from scripts.research.bubble_chamber import s85_native_unknown_audit_v2 as n85


def test_no_native_gt_or_physics_truth_input_to_audit():
    assert tuple(inspect.signature(n85.audit).parameters)==('visualmetro','photo_tiles','models_path','cal_path')
    assert 'physical_auto_joins_authorized' in inspect.getsource(n85.audit)
    assert 'synthetic_nodes' not in inspect.signature(n85.audit).parameters


def test_missing_private_cal_prevents_any_native_claim(tmp_path):
    with pytest.raises(FileNotFoundError,match='TRAIN and CAL'):
        n85.audit(tmp_path,tmp_path,tmp_path/'train.json',tmp_path/'cal.json')
