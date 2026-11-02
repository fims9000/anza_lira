"""CT28 five-patient 3D CCTA development-validation receipts, synthetic only."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import ccta_3d_val_cohort as cohort
from ccta_3d_val_cohort import summarize_validation,write_validation_summary


def write(path,record):
    path.write_text(json.dumps(record),encoding="utf-8")


def create_patient(root,patient,protocol,delta):
    target=root/f"patient_{patient}"
    target.mkdir()
    broken=target/"broken_mask.nii.gz"
    repaired=target/"repaired.nii.gz"
    broken.write_bytes(b"synthetic broken mask"+str(patient).encode())
    repaired.write_bytes(b"synthetic repaired mask"+str(patient).encode())
    write(target/"pilot_manifest.json",{
        "status":"complete_frozen_ct28_development_validation_pilot",
        "patient_id":patient,"split":"val",
        "protocol_sha256":cohort.sha(protocol),
        "held_out_test_accessed":False,
        "ground_truth_used_for_inference":False,
        "frozen_selected_model_sha256":"fake_K10_SHA",
    })
    write(target/"decisions.json",{
        "patient_label":str(patient),
        "scene_count":10,"accepted_decisions":1,"added_voxels":2,
    })
    write(target/"posthoc_metrics.json",{
        "patient_id":str(patient),
        "before_sha256":cohort.sha(broken),
        "after_sha256":cohort.sha(repaired),
        "dice_before":.70,"dice_after":.70+delta,"dice_delta":delta,
        "cldice_before":.60,"cldice_after":.60+delta,"cldice_delta":delta,
        "added_voxels":2,"reference_unsupported_added_voxels":1,
        "gt_inconsistent_new_component_pairs":1,
        "gt_unassigned_new_component_pairs":0,
    })
    return target


def patch_freeze(monkeypatch,tmp_path):
    freeze=tmp_path/"frozen.json"
    freeze.write_text('{"fake":"synthetic"}')
    monkeypatch.setattr(cohort,"verify_validation_freeze",lambda _: {
        "selected_K10_SHA256":"fake_K10_SHA",
    })
    return freeze


def test_fixed_five_patient_development_val_aggregation(tmp_path,monkeypatch):
    freeze=patch_freeze(monkeypatch,tmp_path)
    for i,pid in enumerate((957,961,965,966,974)):
        create_patient(tmp_path,pid,freeze,.01*(i+1))
    report,rows=summarize_validation(
        root=tmp_path,protocol=freeze,bootstrap=102,seed=7,
    )
    assert report["status"]=="complete_five_subject_development_validation"
    assert report["patients_completed"]==5
    assert report["total_repairs"]==5
    assert report["total_gt_inconsistent_component_pairs"]==5
    assert report["patient_macro"]["dice_delta"]["mean_patient_delta"]==pytest.approx(.03)
    assert report["held_out_test_accessed"] is False
    write_validation_summary(tmp_path,report,rows)
    assert (tmp_path/"validation_cohort_results.md").exists()
    with pytest.raises(FileExistsError):
        write_validation_summary(tmp_path,report,rows)


def test_missing_or_failed_subjects_cannot_be_presented_as_complete(tmp_path,monkeypatch):
    freeze=patch_freeze(monkeypatch,tmp_path)
    create_patient(tmp_path,957,freeze,.02)
    fail=tmp_path/"patient_961"
    fail.mkdir()
    write(fail/"pilot_failure.json",{"error_type":"ValueError","error":"no gaps"})
    report,rows=summarize_validation(root=tmp_path,protocol=freeze,bootstrap=100)
    assert report["status"]=="partial_or_subset_development_validation"
    assert (report["patients_completed"],report["patients_failed"],
            report["patients_missing"])==(1,1,3)


def test_heldout_or_tampered_mask_provenance_is_blocked(tmp_path,monkeypatch):
    freeze=patch_freeze(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match="Only five"):
        summarize_validation(root=tmp_path,protocol=freeze,patient_ids=[954],bootstrap=100)
    path=create_patient(tmp_path,957,freeze,.02)
    (path/"broken_mask.nii.gz").write_bytes(b"changed after evidence")
    with pytest.raises(ValueError,match="provenance"):
        summarize_validation(root=tmp_path,protocol=freeze,patient_ids=[957],bootstrap=100)
