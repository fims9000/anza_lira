"""Real saved CCTA features: repeated sham negative-control smoke (test closed)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_ct_sham_robustness import run_repeated_sham,describe
from ct28_ct_signal_benchmark import DEFAULT_SOURCE,sha


def test_three_predeclared_patientwise_shams_use_same_real_CCTA_feature_source(tmp_path):
    output=tmp_path/"sham"
    result=run_repeated_sham(
        input_csv=DEFAULT_SOURCE,output_dir=output,
        repetitions=3,max_iter=10,seed=20261102,
    )
    assert result["source_feature_sha256"]==sha(DEFAULT_SOURCE)
    assert result["held_out_test_accessed"] is False
    assert result["val_patients"]==[957,961,965,966,974]
    assert len(result["train_patients"])==17
    assert result["sham_csv_sha256"]==sha(output/"sham_repetitions.csv")
    for kind in ("PAIR","JUNCTION"):
        d=result["results"][kind]
        assert d["patientwise_sham_CT"]["AP"]["repetitions"]==3
        assert set(d["patientwise_sham_CT"]["per_patient_real_minus_sham_AP"])=={
            "957","961","965","966","974"
        }
        assert 0<=d["aligned_real_CT"]["AP"]<=1
        assert d["val_positive"]>0
    with pytest.raises(FileExistsError):
        run_repeated_sham(
            input_csv=DEFAULT_SOURCE,output_dir=output,
            repetitions=3,max_iter=10,
        )


def test_sham_statistic_errors_are_not_silent():
    with pytest.raises(ValueError):
        describe([])
    with pytest.raises(ValueError):
        describe([float("nan")])
    with pytest.raises(ValueError,match="3..200"):
        run_repeated_sham(
            input_csv=DEFAULT_SOURCE,output_dir=Path("/tmp/not-created"),
            repetitions=2,
        )
