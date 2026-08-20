from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _validator_module():
    path = ROOT / "scripts" / "validate_anza2_phase3c_a.py"
    spec = importlib.util.spec_from_file_location("validate_anza2_phase3c_a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase3c_a_validator_accepts_frozen_forensic_artifacts():
    result = _validator_module().validate()
    assert result["status"] == "PASS"
    assert result["research_status"] == "PHASE3C_A_FORENSIC_PASS_ROOT_CAUSE_MEMBERSHIP_LEARNING"
    assert result["root_cause"]["rc_code"] == "RC1"
    assert result["repair_allowed"] is True
    assert result["repair_performed"] is False


def test_phase3c_a_validator_preserves_all_data_locks():
    result = _validator_module().validate()
    assert result["training_performed"] is False
    assert result["confirm_opened"] is False
    assert result["cracks_data_accessed"] is False
    assert result["expert_data_accessed"] is False
