"""Tests for train-only anatomical segment-ID evidence and physical coordinates."""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import audit_ct28_anatomical_segments as audit


def _fixture(tmp_path):
    scenes=tmp_path/"scenes"
    scenes.mkdir()
    sid="fake_train_953"
    pd.DataFrame([{
        "scene_id":sid,"scan_id":953,"split":"train"
    }]).to_csv(scenes/"scene_plan_train_val.csv",index=False)
    locations=[]
    for i in range(3):
        locations.append({
            "scene_id":sid,"endpoint_id":f"E{i}",
            "x_lps":-15.-i,"y_lps":-14.,"z_lps":14.,
        })
    pd.DataFrame(locations).to_csv(scenes/"scene_endpoints_train_val.csv",index=False)
    pd.DataFrame([
        {"scene_id":sid,"candidate_id":"p0","candidate_type":"PAIR",
         "endpoint_ids":"E0|E1","y":1},
        {"scene_id":sid,"candidate_id":"p1","candidate_type":"PAIR",
         "endpoint_ids":"E1|E2","y":0},
        {"scene_id":sid,"candidate_id":"j0","candidate_type":"JUNCTION",
         "endpoint_ids":"E0|E1|E2","y":0},
    ]).to_csv(scenes/"scene_candidates_train_val.csv",index=False)
    label_path=tmp_path/"segmentations"
    label_path.mkdir()
    labels=np.zeros((30,30,30),dtype=np.int16)
    labels[15,14,14]=1
    labels[16,14,14]=1
    labels[17,14,14]=2
    nib.save(nib.Nifti1Image(labels,np.eye(4)),str(label_path/"953.coronary.nii.gz"))
    return scenes,label_path


def test_anatomical_segment_audit_reports_mixed_real_branches(tmp_path,monkeypatch):
    scenes,labels=_fixture(tmp_path)
    def fake_verified_ct(patient,archive):
        assert patient==953
        return np.zeros((30,30,30),dtype=np.float32),np.eye(4)
    monkeypatch.setattr(audit,"load_verified_training_or_validation_ct",fake_verified_ct)
    out=tmp_path/"output"/"branches.json"
    result=audit.audit(
        scene_dir=scenes,archive_dir=tmp_path,
        segmentations=labels,patients=[953],output=out,
    )
    assert result["held_out_test_accessed"] is False
    assert result["validation_patient_accessed"] is False
    assert result["groups"]["PAIR/y=1"]["same_segment_fraction_among_labeled"]==1.
    assert result["groups"]["PAIR/y=0"]["same_segment_fraction_among_labeled"]==0.
    assert result["groups"]["JUNCTION/y=0"]["same_segment_fraction_among_labeled"]==0.
    records=pd.read_csv(out.with_suffix(".csv"))
    assert records.shape[0]==3 and records.segments.tolist()==["1|1","1|2","1|1|2"]
    with pytest.raises(FileExistsError):
        audit.audit(scene_dir=scenes,archive_dir=tmp_path,
                    segmentations=labels,patients=[953],output=out)


@pytest.mark.parametrize("pid",[957,966,980,984])
def test_branch_audit_rejects_val_and_test_before_loading_files(tmp_path,pid):
    with pytest.raises(ValueError,match="Validation or held-out"):
        audit.audit(scene_dir=tmp_path,archive_dir=tmp_path,
                    segmentations=tmp_path,patients=[pid],
                    output=tmp_path/"out.json")
