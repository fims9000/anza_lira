"""Actual committed CT28 count-only generator-shortcut negative control."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_cardinality_benchmark import load_frozen,compute_baseline,write_report,SCENES


def test_count_only_negative_control_on_actual_committed_ct28():
    train,val=load_frozen()
    report,rows=compute_baseline(train,val,bootstrap=201,seed=18)
    assert (len(train),len(val))==(1036,368)
    assert (report["validation_correct"],report["validation_scenes"])==(270,368)
    assert report["validation_accuracy"]==pytest.approx(270/368)
    assert report["validation_patterns_seen_in_train"]==367
    assert report["validation_patterns_unseen_in_train"]==1
    assert report["training_count_patterns"]==28
    assert report["fallback_relation"]=="NONE"
    assert set(report["per_patient"])=={"957","961","965","966","974"}
    assert sum(row["correct"] for row in rows)==270
    assert report["held_out_test_accessed"] is False
    assert 0<report["validation_macro_f1"]<1


def test_immutable_count_baseline_report_and_test_patient_firewall(tmp_path):
    train,val=load_frozen()
    report,rows=compute_baseline(train,val,bootstrap=100)
    target=tmp_path/"baseline"
    write_report(target,report,rows)
    assert len(pd.read_csv(target/"count_only_val_predictions.csv"))==368
    saved=json.loads((target/"count_only_baseline.json").read_text())
    assert saved["validation_correct"]==270
    with pytest.raises(FileExistsError):
        write_report(target,report,rows)
    changed=pd.read_csv(SCENES)
    changed.loc[changed.split.eq("val"),"scan_id"]=954
    bad=tmp_path/"bad_scene_plan.csv"
    changed.to_csv(bad,index=False)
    with pytest.raises(ValueError,match="Patient split"):
        load_frozen(bad)
