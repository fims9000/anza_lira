"""Read-only CCTA medical inputs preflight exercises paths, not patient data."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                       "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_preflight import assess_inputs,REQUIRED_CT28_FILES
from extract_ct28_scene_candidate_features import EXPECTED_PATIENT_IDS


def make_paths(tmp_path):
    scene=tmp_path/"scene"
    scene.mkdir()
    for f in REQUIRED_CT28_FILES:
        (scene/f).write_bytes(b"synthetic_saved_scene")
    archive=tmp_path/"archive"
    archive.mkdir()
    for f in ("801-1000.z04","801-1000.zip"):
        (archive/f).write_bytes(b"not_an_actual_Imaging_archive")
    mask=tmp_path/"masks"
    mask.mkdir()
    for pid in EXPECTED_PATIENT_IDS["train"]:
        (mask/f"{pid}.coronary.nii.gz").write_bytes(b"synthetic_mask")
    model=tmp_path/"fake_K10.joblib"
    model.write_bytes(b"synthetic")
    path=tmp_path/"frozen_path.json"
    path.write_text("{}")
    return scene,archive,mask,model,path


def test_preflight_rejects_missing_validation_without_touching_heldout(tmp_path):
    scene,archive,masks,model,path=make_paths(tmp_path)
    args=dict(
        scene_dir=scene,archive_dir=archive,
        reference_dir=masks,k10_pack=model,
        path_protocol=path,min_free_gb=0.,
        expected_k10_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    ready=assess_inputs(**args)
    assert ready["status"]=="all_required_local_sources_present"
    assert ready["patient_count_train"]==17
    assert ready["patient_count_val_required"]==0
    assert not ready["issues"]
    dev=assess_inputs(**args,require_development_val=True)
    assert dev["status"]=="missing_or_invalid_local_inputs"
    assert dev["patient_count_val_required"]==5
    assert len(dev["issues"])==5
    assert all(str(p).startswith("reference_mask_val:") for p in dev["issues"])
    assert dev["held_out_test_accessed"] is False
    for item in dev["checked_inputs"]:
        assert not any(str(pid)+".coronary" in item["path"]
                       for pid in (954,958,972,973,980,984))


def test_invalid_selected_K10_sha_detected_from_read_only_bytes(tmp_path):
    scene,archive,masks,model,path=make_paths(tmp_path)
    report=assess_inputs(
        scene_dir=scene,archive_dir=archive,
        reference_dir=masks,k10_pack=model,path_protocol=path,
        min_free_gb=0.,expected_k10_sha256="f"*64,
    )
    assert "selected_K10_model_SHA_mismatch" in report["issues"]


def test_missing_selected_model_and_archive_are_actionable(tmp_path):
    scene,archive,masks,model,path=make_paths(tmp_path)
    model.unlink()
    (archive/"801-1000.z04").unlink()
    result=assess_inputs(
        scene_dir=scene,archive_dir=archive,
        reference_dir=masks,k10_pack=model,path_protocol=path,
        min_free_gb=0.,
    )
    assert "selected_K10_15_local_model_pack" in result["issues"]
    assert "ImageCAS_CT_archive:801-1000.z04" in result["issues"]
