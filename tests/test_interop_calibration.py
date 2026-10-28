from pathlib import Path
from dataclasses import asdict
from hashlib import sha256
import json
import numpy as np
import pytest
from bubble_chamber.interop_calibration import CalibrationReceipt


def _fixture(tmp_path):
    model=tmp_path/'private_fake_model.bin';model.write_bytes(b'fake-weights-TOY-NOT-PHYSICS')
    cal=tmp_path/'fake_cal.json';cal.write_text('toy-cal')
    x=CalibrationReceipt('opencv-lsd-v1','synthetic-QA',sha256(model.read_bytes()).hexdigest(),
                         sha256(cal.read_bytes()).hexdigest(),'TOY_CAL','toy-synthetic',
                         .75,.04,'port_greedy','toy_only')
    return x,model,cal


def test_receipt_fails_closed_on_detector_swap_or_replaced_model(tmp_path):
    x,model,cal=_fixture(tmp_path)
    f=lambda rows:np.zeros(len(rows))
    b=x.binding(detector_id='opencv-lsd-v1',model_file=model,calibration_file=cal,scorer=f)
    assert b.trained_for_detector=='opencv-lsd-v1' and b.strategy=='port_greedy'
    assert b.source_cal_id=='TOY_CAL'
    with pytest.raises(ValueError,match='identity'):
        x.binding(detector_id='hessian-ridge-v1',model_file=model,calibration_file=cal,scorer=f)
    cal.write_text('changed')
    with pytest.raises(ValueError,match='Source-CAL'):
        x.binding(detector_id='opencv-lsd-v1',model_file=model,calibration_file=cal,scorer=f)
    cal.write_text('toy-cal')
    model.write_bytes(b'other')
    with pytest.raises(ValueError,match='checksum'):
        x.binding(detector_id='opencv-lsd-v1',model_file=model,calibration_file=cal,scorer=f)


def test_receipt_roundtrip_and_privacy(tmp_path):
    x,_,_=_fixture(tmp_path)
    p=tmp_path/'receipt.json';p.write_text(json.dumps(asdict(x)))
    loaded=CalibrationReceipt.from_file(p)
    assert loaded==x
    assert loaded.public_dict()['physical_auto_joins_authorized'] is False
    j=json.loads(p.read_text());j['training_domain_id']='changed';j['unexpected_GT']=1
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='EXACTLY'):
        CalibrationReceipt.from_file(p)


def test_reject_bad_checksums_and_bad_validation(tmp_path):
    x,_,_=_fixture(tmp_path);values=asdict(x)
    values['model_sha256']='missing'
    with pytest.raises(ValueError):CalibrationReceipt(**values)
    values=asdict(x);values['validation_level']='physical_proven_in_chat'
    with pytest.raises(ValueError):CalibrationReceipt(**values)
