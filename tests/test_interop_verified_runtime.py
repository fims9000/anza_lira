import hashlib
from pathlib import Path
import numpy as np
import pytest
from bubble_chamber.interop_calibration import CalibrationReceipt
from bubble_chamber.interop_verified_runtime import process_verified


class ToyDetector:
    detector_id='shadow-detector'
    def detect(self,gray):
        assert not gray.flags.writeable
        return []


def fixture(tmp_path,level='synthetic_on_real_photo'):
    model=tmp_path/'model.private';model.write_bytes(b'safe-fake-model-sha-qa')
    cal=tmp_path/'cal.private';cal.write_bytes(b'source-cal-fake-qa')
    rec=CalibrationReceipt('shadow-detector','pair-scorer-v1',
              hashlib.sha256(model.read_bytes()).hexdigest(),
              hashlib.sha256(cal.read_bytes()).hexdigest(),
              'SOURCE_CAL_2026','lab-A-photo-domain',.87,.04,'port_greedy',level)
    return rec,model,cal


def call(gray,rec,model,cal,**kwargs):
    return process_verified(gray,ToyDetector(),receipt=rec,model_file=model,
                 source_cal_file=cal,score_pairs=lambda rows:np.zeros((len(rows),)),
                 expected_training_domain='lab-A-photo-domain',**kwargs)


def test_verify_contract_source_crop_and_readonly_prediction(tmp_path):
    receipt,model,cal=fixture(tmp_path)
    source=np.full((192,192),200,np.uint8)
    result=call(source,receipt,model,cal)
    assert result.detector_id=='shadow-detector'
    assert result.solver_strategy=='port_greedy'
    assert result.source_cal_id=='SOURCE_CAL_2026'
    assert not result.joins and result.as_public_dict()['native_physics_gt_count']==0


def test_toy_level_never_passes_research_gate(tmp_path):
    receipt,model,cal=fixture(tmp_path,'toy_only')
    with pytest.raises(ValueError,match='toy-only'):
        call(np.zeros((192,192),np.uint8),receipt,model,cal)
    # Deliberate toy QA can explicitly lower its own threshold, but never auto-join.
    toy=call(np.zeros((192,192),np.uint8),receipt,model,cal,minimum_evidence='toy_only')
    assert not toy.physical_auto_joins_authorized


def test_training_domain_detector_identity_and_model_sha_cannot_be_mixed(tmp_path):
    receipt,model,cal=fixture(tmp_path)
    with pytest.raises(ValueError,match='Wrong source'):
        process_verified(np.zeros((192,192),np.uint8),ToyDetector(),receipt=receipt,
              model_file=model,source_cal_file=cal,score_pairs=lambda rows:np.zeros(len(rows)),
              expected_training_domain='different-experiment')
    model.write_bytes(b'drift')
    with pytest.raises(ValueError,match='checksum'):
        call(np.zeros((192,192),np.uint8),receipt,model,cal)
    model.write_bytes(b'safe-fake-model-sha-qa')
    cal.write_bytes(b'drift')
    with pytest.raises(ValueError,match='Source-CAL'):
        call(np.zeros((192,192),np.uint8),receipt,model,cal)
