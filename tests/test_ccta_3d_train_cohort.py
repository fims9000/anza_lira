"""Patient-level train CCTA cohort statistics from synthetic JSON receipts."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_train_cohort import summarize_cohort,write_report


def fixture_patient(root,patient,delta=.2,pack="fixed_k10"):
    folder=root/f"patient_{patient}"
    folder.mkdir()
    manifest={
        "status":"completed_experimental_train_pilot",
        "patient_id":patient,
        "mode":"k10",
        "source_kind":"controlled_gap_from_reference_not_natural_segmentation_failure",
        "ground_truth_used_in_inference":False,
        "model_pack_sha256":pack,
        "gap_generation":{"gaps":3,"gap_radius_mm":1.5,"seed":20261102},
    }
    decisions={
        "patient_label":str(patient),
        "scene_count":6,"endpoint_count":10,
        "radius_mm":0.,
        "max_new_voxels_per_decision":250,
        "path_slack_fraction":.05,
        "accepted_decisions":2,
        "added_voxels":10,
        "abstention_reasons":{"K10_abstain":4},
    }
    metrics={
        "patient_id":str(patient),
        "truth_used_for_inference":False,
        "added_voxels":10,
        "reference_unsupported_added_voxels":3,
        "dice_before":.7,
        "dice_after":.7+delta,
        "dice_delta":delta,
        "cldice_before":.6,
        "cldice_after":.6+delta,
        "cldice_delta":delta,
    }
    for name,record in (("pilot_manifest.json",manifest),
                        ("decisions.json",decisions),
                        ("posthoc_metrics.json",metrics)):
        (folder/name).write_text(json.dumps(record))


def test_patient_paired_bootstrap_and_markdown(tmp_path):
    fixture_patient(tmp_path,953,.1)
    fixture_patient(tmp_path,955,.2)
    fixture_patient(tmp_path,956,.3)
    summary,patients=summarize_cohort(root=tmp_path,patients=[953,955,956],
                                      bootstrap=501,seed=7)
    assert summary["status"]=="complete_train_cohort"
    assert summary["patient_count_completed"]==3
    assert summary["total_accepted_repairs"]==6
    assert summary["total_added_voxels"]==30
    assert summary["total_false_added_voxels"]==9
    assert summary["reasons"]["abstain:K10_abstain"]==12
    d=summary["paired_metrics_patient_macro"]["dice_delta"]
    assert d["mean"]==pytest.approx(.2)
    assert d["exploratory_patient_bootstrap_95pct"][0]<=.2
    assert d["exploratory_patient_bootstrap_95pct"][1]>=.2
    write_report(tmp_path,summary,patients)
    assert (tmp_path/"train_cohort_summary.json").is_file()
    assert (tmp_path/"train_cohort_patients.csv").is_file()
    assert "train-only" in (tmp_path/"train_cohort_results.md").read_text()
    with pytest.raises(FileExistsError):
        write_report(tmp_path,summary,patients)


def test_missing_and_failed_patients_must_remain_explicit(tmp_path):
    fixture_patient(tmp_path,953,.1)
    failure=tmp_path/"patient_955"
    failure.mkdir()
    (failure/"pilot_failure.json").write_text(json.dumps({
        "status":"failed_experimental_train_pilot",
        "error_type":"ValueError","error":"No sufficiently long vessel",
    }))
    report,rows=summarize_cohort(root=tmp_path,patients=[953,955,956],bootstrap=100)
    assert report["status"]=="partial_train_cohort"
    assert (report["patient_count_completed"],report["patient_count_failed"],
            report["patient_count_missing"])==(1,1,1)
    assert {x["status"] for x in rows}=={"complete","failed","missing"}


def test_cannot_mix_models_or_train_and_test(tmp_path):
    fixture_patient(tmp_path,953,.1,pack="pack_a")
    fixture_patient(tmp_path,955,.2,pack="pack_b")
    with pytest.raises(ValueError,match="distinct K10"):
        summarize_cohort(root=tmp_path,patients=[953,955],bootstrap=100)
    with pytest.raises(ValueError,match="train-only"):
        summarize_cohort(root=tmp_path,patients=[953,954],bootstrap=100)


def test_invalid_ground_truth_leak_evidence_fails(tmp_path):
    fixture_patient(tmp_path,953)
    p=tmp_path/"patient_953"/"pilot_manifest.json"
    j=json.loads(p.read_text())
    j["ground_truth_used_in_inference"]=True
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="leaked"):
        summarize_cohort(root=tmp_path,patients=[953],bootstrap=100)
