import json
from hashlib import sha256
from pathlib import Path
import pytest
from scripts.research.bubble_chamber.s63_package_private_demo import run,SCHEMA


def test_reject_changed_calibration(tmp_path):
    a={'schema':SCHEMA,'phase':'cal','cal_only_decision':{'chosen_on_CAL_only':'junction_less'}}
    b={'schema':SCHEMA,'phase':'test','calibration_sha256':'x','selected_method':'junction_less','total':{}}
    ca=tmp_path/'cal.json';te=tmp_path/'test.json'
    ca.write_text(json.dumps(a));te.write_text(json.dumps(b))
    with pytest.raises(ValueError,match='different CAL'):
        run(tmp_path,ca,te,tmp_path/'out',tmp_path/'private.zip')


def test_negative_effect_is_not_physical_claim():
    source=Path(__file__).resolve().parents[1]/'scripts/research/bubble_chamber/s63_package_private_demo.py'
    text=source.read_text()
    assert 'no expert' not in text.lower() or 'native_GT' in text
    assert "'default_physics_auto_joins':0" in text
    assert 'НЕ доказывают' in text
