import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "validate_anza_s_oracle.py"
    spec = importlib.util.spec_from_file_location("validate_anza_s_oracle", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_anza_s_validator_accepts_frozen_oracle_artifacts():
    result = _module().validate()
    assert result["status"] == "PASS"
    assert result["research_status"] in {"ANZA_S_ORACLE_GATE_A_PASS", "FINAL_STOP_ANOSOV_DYNAMICAL_ARCHITECTURE"}


def test_anza_s_validator_keeps_training_and_data_locks_closed():
    result = _module().validate()
    assert result["training_performed"] is False
    assert result["confirm_opened"] is False
    assert result["cracks_data_accessed"] is False
    assert result["expert_data_accessed"] is False
