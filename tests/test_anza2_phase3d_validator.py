import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "validate_anza2_phase3d_ab.py"
    spec = importlib.util.spec_from_file_location("validate_anza2_phase3d_ab", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_phase3d_validator_accepts_frozen_artifacts_when_present():
    result = _module().validate()
    assert result["status"] == "PASS"
    assert result["research_status"] in {
        "PHASE3D_ORACLE_MODE_STATE_PASS", "FINAL_STOP_MODE_STATE_ORACLE_NO_VALUE",
    }


def test_phase3d_validator_preserves_all_locks():
    result = _module().validate()
    assert result["training_performed"] is False
    assert result["confirm_evaluation_opened"] is False
    assert result["cracks_data_accessed"] is False
    assert result["expert_data_accessed"] is False
