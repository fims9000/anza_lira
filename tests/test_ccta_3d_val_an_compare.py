"""Matched five-patient CNN/ANZA/K10 CCTA comparison without medical volume."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import ccta_3d_val_an_compare as compare


def write(path,object):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(object))


def fake_patient(pilot_root,result_root,patient,freeze,an_delta,relation_training="clean"):
    pilot=pilot_root/f"patient_{patient}"
    pilot.mkdir(parents=True,exist_ok=True)
    (pilot/"ct.nii.gz").write_bytes(f"synthetic_ct_{patient}".encode())
    (pilot/"broken_mask.nii.gz").write_bytes(f"synthetic_broken_{patient}".encode())
    baseline={
        "patient_id":str(patient),
        "before_sha256":compare.sha(pilot/"broken_mask.nii.gz"),
        "reference_sha256":"same_GT_patient_"+str(patient),
        "dice_delta":.06,
        "cldice_delta":.05,
        "gt_inconsistent_new_component_pairs":0,
        "reference_unsupported_added_voxels":1,
    }
    write(pilot/"posthoc_metrics.json",baseline)
    write(pilot/"pilot_manifest.json",{
        "status":"complete_frozen_ct28_development_validation_pilot",
        "patient_id":patient,
        "split":"val",
        "protocol_sha256":compare.sha(freeze),
        "held_out_test_accessed":False,
        "ground_truth_used_for_inference":False,
        "ct_sha256":compare.sha(pilot/"ct.nii.gz"),
    })
    for model,delta in (("conv3d",.03),("anza3d",an_delta)):
        out=result_root/f"patient_{patient}"/model
        out.mkdir(parents=True,exist_ok=True)
        (out/"repaired.nii.gz").write_bytes((model+str(patient)).encode())
        write(out/"result_manifest.json",{
            "status":"complete_exploratory_neural_CT28_development_validation",
            "patient_id":patient,
            "variant":model,
            "pair_identity_source":"frozen_fusion",
            "relation_train_protocol":compare._relation_protocol(relation_training),
            "freeze_sha256":compare.sha(freeze),
        })
        write(out/"posthoc_metrics.json",{
            **baseline,
            "dice_delta":delta,
            "cldice_delta":delta,
            "after_sha256":compare.sha(out/"repaired.nii.gz"),
        })


def freeze_stub(tmp_path,monkeypatch):
    file=tmp_path/"freeze.json"
    file.write_text("{}")
    original=compare.verify_validation_freeze
    def stub(_):
        return {"expected_validation_CT_SHA256":{
            str(patient):compare.sha(tmp_path/"pilots"/f"patient_{patient}"/"ct.nii.gz")
            for patient in (957,961,965,966,974)
            if (tmp_path/"pilots"/f"patient_{patient}"/"ct.nii.gz").exists()
        }}
    monkeypatch.setattr(compare,"verify_validation_freeze",stub)
    return file


def test_five_patient_paired_delta_not_confounded_by_different_masks(tmp_path,monkeypatch):
    freeze=freeze_stub(tmp_path,monkeypatch)
    p=tmp_path/"pilots"
    o=tmp_path/"neural"
    for i,patient in enumerate((957,961,965,966,974)):
        fake_patient(p,o,patient,freeze,.02+.01*i)
    report,rows=compare.summarize_matched_validation(
        freeze_path=freeze,pilot_root=p,output_root=o,
        bootstrap=151,seed=42,
    )
    assert report["status"]=="complete_five_patient_matched_CCTA_dev_val"
    assert report["patient_completed"]==5
    assert report["paired_ANZA_minus_CNN"]["ANZA_minus_CNN_cldice_delta"]["mean_paired_patient_delta"]==pytest.approx(.01)
    assert report["held_out_test_accessed"] is False
    compare.write_matched(o,report,rows)
    assert (o/"matched_val_comparison.md").is_file()
    assert len(json.loads((o/"matched_val_comparison.json").read_text())["per_patient"])==5
    with pytest.raises(FileExistsError):
        compare.write_matched(o,report,rows)


def test_missing_neural_result_cannot_be_reported_as_complete(tmp_path,monkeypatch):
    freeze=freeze_stub(tmp_path,monkeypatch)
    p=tmp_path/"pilots"
    o=tmp_path/"neural"
    fake_patient(p,o,957,freeze,.05)
    (o/"patient_957"/"anza3d"/"result_manifest.json").unlink()
    report,rows=compare.summarize_matched_validation(
        freeze_path=freeze,pilot_root=p,output_root=o,
        bootstrap=100,
    )
    assert report["status"]=="partial_CCTA_development_validation_neural_comparison"
    assert report["patient_completed"]==0
    assert report["patient_failed_or_missing"]==5


def test_test_subject_never_passes_to_volume_loader(tmp_path):
    with pytest.raises(ValueError,match="Held-out"):
        compare.run_neural_val_patient(
            patient=954,variant="anza3d",freeze_path=tmp_path/"freeze.json",
            pilot_root=tmp_path/"absent",study_root=tmp_path/"absent",
            output_root=tmp_path/"absent",device="cpu",
        )


def test_experimental_pair_identity_source_cannot_mix_in_same_val_report(tmp_path,monkeypatch):
    freeze=freeze_stub(tmp_path,monkeypatch)
    baseline=tmp_path/"pilots"
    neural=tmp_path/"neural"
    for patient in (957,961,965,966,974):
        fake_patient(baseline,neural,patient,freeze,.07)
    with pytest.raises(ValueError,match="Neural CCTA validation receipt or freeze mismatch"):
        compare.summarize_matched_validation(
            freeze_path=freeze,pilot_root=baseline,output_root=neural,
            bootstrap=100,pair_identity_source="neural_pair",
        )


def test_matched_five_patient_augmented_relation_mode_is_explicit(tmp_path,monkeypatch):
    freeze=freeze_stub(tmp_path,monkeypatch)
    pilot=tmp_path/"pilots"
    results=tmp_path/"neural"
    for patient in (957,961,965,966,974):
        fake_patient(pilot,results,patient,freeze,.04,relation_training="augmented_15")
    report,rows=compare.summarize_matched_validation(
        freeze_path=freeze,pilot_root=pilot,output_root=results,
        bootstrap=100,relation_training="augmented_15",
    )
    assert report["status"]=="complete_five_patient_matched_CCTA_dev_val"
    assert report["relation_train_protocol"]==compare._relation_protocol("augmented_15")
    assert any("15 patient-OOF perturbation copies" in text for text in report["limitations"])
    assert len(rows)==5
    with pytest.raises(ValueError,match="receipt or freeze"):
        compare.summarize_matched_validation(
            freeze_path=freeze,pilot_root=pilot,output_root=results,
            bootstrap=100,relation_training="clean",
        )


def test_validation_rejects_unknown_relation_training_mode_before_any_CT(tmp_path):
    with pytest.raises(ValueError,match="Expected clean or augmented_15"):
        compare.run_neural_val_patient(
            patient=957,variant="anza3d",
            freeze_path=tmp_path/"absent_freeze.json",
            pilot_root=tmp_path,study_root=tmp_path,
            output_root=tmp_path,device="cpu",
            relation_training="unverified",
        )
