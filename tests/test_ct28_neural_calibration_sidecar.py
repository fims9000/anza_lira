"""Clinical CT28 calibration sidecar: train fit, val prediction and SHA isolation."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from audit_ct28_neural_calibration import audit
from ccta_3d_neural_calibration import load_calibrator


def fixture(tmp_path):
    scene_dir=tmp_path/"scenes"
    oof_dir=tmp_path/"oof"
    scene_dir.mkdir()
    oof_dir.mkdir()
    plan,candidate,scores=[],[],[]
    for pid,split in ((953,"train"),(955,"train"),(956,"train"),(957,"val")):
        for j in range(6):
            scene=f"synthetic_patient_{pid}_scene_{j}"
            plan.append({"scene_id":scene,"scan_id":pid,"split":split})
            label=int(j in (0,3))
            candidate.append({
                "scene_id":scene,"candidate_id":f"JUNCTION:E0+E1+E2_{j}",
                "candidate_type":"JUNCTION","y":label,
            })
            scores.append({
                "scene_id":scene,"candidate_id":f"JUNCTION:E0+E1+E2_{j}",
                "candidate_type":"JUNCTION","scan_id":pid,
                "split":split,"oof_fold":j%2 if split=="train" else -1,
                "score":.35+.18*label+.01*(j%3),
            })
    pd.DataFrame(plan).to_csv(scene_dir/"scene_plan_train_val.csv",index=False)
    pd.DataFrame(candidate).to_csv(scene_dir/"scene_candidates_train_val.csv",index=False)
    scores_file=oof_dir/"JUNCTION_anza3d_sep_structure_oof_val.csv"
    pd.DataFrame(scores).to_csv(scores_file,index=False)
    checksum=hashlib.sha256(scores_file.read_bytes()).hexdigest()
    (oof_dir/"oof_protocol.json").write_text(json.dumps({
        "status":"train_patient_OOF_experimental_CT_evidence_not_promoted_K10",
        "held_out_test_accessed":False,
        "variant":"anza3d_sep_structure",
        "candidate_type":"JUNCTION",
        "score_sha256":checksum,
    }))
    return scene_dir,oof_dir


def test_sidecar_calibrates_train_patient_crossfit_and_scored_val_without_val_fit(tmp_path):
    scene,oof=fixture(tmp_path)
    receipt=audit(
        kind="JUNCTION",variant="anza3d_sep_structure",oof_dir=oof,
        scenes_dir=scene,out_dir=tmp_path/"cal",
        expected_train=frozenset({953,955,956}),expected_val=frozenset({957}),
    )
    assert receipt["status"]=="train_only_neural_calibration_sidecar_not_activated"
    assert receipt["n_train_candidates"]==18
    assert receipt["n_val_candidates_scored_without_labels"]==6
    assert receipt["held_out_CT28_test_accessed"] is False
    assert receipt["validation_labels_used_for_fit"] is False
    assert len(receipt["patient_crossfit_provenance"])==3
    for row in receipt["patient_crossfit_provenance"]:
        assert row["held_patient_id"] not in row["calibration_train_ids"]
    final=load_calibrator(tmp_path/"cal"/"train_only_platt.json")
    derived=pd.read_csv(tmp_path/"cal"/"JUNCTION_anza3d_sep_structure_calibrated_sidecar.csv")
    val=derived[derived.split.eq("val")]
    assert np.allclose(val.score,final.predict(val.raw_score.to_numpy()))
    assert set(derived.calibration_fit)=={
        "all_other_train_patients","all_train_patients"
    }
    with pytest.raises(FileExistsError):
        audit(kind="JUNCTION",variant="anza3d_sep_structure",oof_dir=oof,
              scenes_dir=scene,out_dir=tmp_path/"cal",
              expected_train=frozenset({953,955,956}),expected_val=frozenset({957}))


def test_validation_labels_cannot_change_calibration_scale(tmp_path):
    scene,oof=fixture(tmp_path)
    args=dict(kind="JUNCTION",variant="anza3d_sep_structure",oof_dir=oof,
              scenes_dir=scene,expected_train=frozenset({953,955,956}),
              expected_val=frozenset({957}))
    audit(out_dir=tmp_path/"first",**args)
    original=(tmp_path/"first"/"train_only_platt.json").read_bytes()
    predictions=(tmp_path/"first"/"JUNCTION_anza3d_sep_structure_calibrated_sidecar.csv").read_bytes()
    table=pd.read_csv(scene/"scene_candidates_train_val.csv")
    val=table.scene_id.str.contains("_957_")
    table.loc[val,"y"]=1-table.loc[val,"y"]
    table.to_csv(scene/"scene_candidates_train_val.csv",index=False)
    audit(out_dir=tmp_path/"second",**args)
    assert original==(tmp_path/"second"/"train_only_platt.json").read_bytes()
    assert predictions==(tmp_path/"second"/"JUNCTION_anza3d_sep_structure_calibrated_sidecar.csv").read_bytes()


def test_sidecar_blocks_heldout_patient_even_when_mocked_expected_set(tmp_path):
    scene,oof=fixture(tmp_path)
    with pytest.raises(ValueError,match="train and development"):
        audit(kind="JUNCTION",variant="anza3d_sep_structure",oof_dir=oof,
              scenes_dir=scene,out_dir=tmp_path/"new",
              expected_train=frozenset({953,955,956}),
              expected_val=frozenset({980}))
