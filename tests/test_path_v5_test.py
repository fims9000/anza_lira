import copy
import json
from pathlib import Path

import pytest

from path_completion.calibration import _canonical_hash
from synthetic.crossing_trace_bench_v5 import generate_authorized_test_sample_v5, generate_sample_v5


def test_normal_v5_api_keeps_test_locked():
    with pytest.raises(PermissionError, match="LOCKED"):
        generate_sample_v5("test", 0)


def test_authorized_api_rejects_forged_or_mutated_calibration():
    forged = {
        "status": "CALIBRATION_FROZEN",
        "v5_test_samples_opened": 0,
        "old_confirm_used_for_calibration": False,
        "expert_data_accessed": False,
        "protocol": {"v5_test": "LOCKED_UNOPENED", "calibration_stream": "v5 validation"},
        "freeze_sha256": "forged",
    }
    with pytest.raises(PermissionError):
        generate_authorized_test_sample_v5(0, calibration_freeze=forged)
    frozen = json.loads((Path(__file__).resolve().parents[1] / "results/final_practical_cycle/path_calibration/calibration_freeze.json").read_text())
    mutated = copy.deepcopy(frozen)
    mutated["selected_operating_point"]["threshold"] -= 0.1
    with pytest.raises(PermissionError):
        generate_authorized_test_sample_v5(0, calibration_freeze=mutated)


def test_calibration_hash_is_self_consistent_before_test_open():
    frozen = json.loads((Path(__file__).resolve().parents[1] / "results/final_practical_cycle/path_calibration/calibration_freeze.json").read_text())
    core = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    assert frozen["freeze_sha256"] == _canonical_hash(core)
    assert frozen["v5_test_samples_opened"] == 0
