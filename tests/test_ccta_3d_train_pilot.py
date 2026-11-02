"""Full train-only pipeline on synthetic NIfTI, with archive access mocked.

The production script never accepts validation/held-out CT28 patient IDs.
Tests do not access local medical data or deserialize real model weights.
"""
from pathlib import Path
import json
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import ccta_3d_train_pilot as pilot
import ccta_3d_cli
from ccta_3d_k10 import ScenePrediction


def setup_synthetic_pilot(monkeypatch, tmp_path):
    ref = np.zeros((90, 24, 24), dtype=bool)
    ref[8:80, 12, 12] = True
    affine = np.eye(4)
    source = tmp_path / "original_reference.nii.gz"
    nib.save(nib.Nifti1Image(ref.astype(np.uint8), affine), str(source))

    def fake_export(scan_id, archive_dir, out, report_path):
        assert scan_id == 953
        nib.save(nib.Nifti1Image(
            np.full(ref.shape, 250., dtype=np.float32), affine
        ), str(out))
        meta = {"raw_ct_sha256": pilot.digest(out),
                "status": "synthetic_testing_only"}
        report_path.write_text(json.dumps(meta))
        return meta

    monkeypatch.setattr(pilot, "export_training_ct", fake_export)
    return source


def test_one_command_candidate_pilot_persists_all_evidence(monkeypatch, tmp_path):
    original = setup_synthetic_pilot(monkeypatch, tmp_path)
    output = tmp_path / "pilot"
    report = pilot.execute_train_pilot(
        scan_id=953, output_dir=output, reference_path=original,
        archive_dir=tmp_path, model_pack=None, mode="candidates",
        gaps=3, gap_radius_mm=1.1,
        min_component_voxels=4,
    )
    assert report["status"] == "completed_experimental_train_pilot"
    assert report["candidate_scene_count"] > 0
    assert report["candidate_endpoint_count"] >= 5
    assert (output/"ct.nii.gz").exists()
    assert (output/"broken_mask.nii.gz").exists()
    assert (output/"candidates.json").exists()
    assert not (output/"decisions.json").exists()
    assert (output/"pilot_manifest.json").exists()
    assert nib.load(str(output/"broken_mask.nii.gz")).shape == (90,24,24)
    with pytest.raises(FileExistsError):
        pilot.execute_train_pilot(
            scan_id=953, output_dir=output, reference_path=original,
            archive_dir=tmp_path, model_pack=None, mode="candidates",
            gaps=3, gap_radius_mm=1.1, min_component_voxels=4,
        )


class StubK10:
    pack_sha = "synthetic_test_only"

    def predict(self, scene, endpoints, ct, affine):
        pair = next((x for x in scene.candidates if x.kind == "PAIR"), None)
        if pair is None:
            return ScenePrediction(scene.scene_id,"NONE","NONE","NONE",.99,1,False,
                                    None,None,0,0,0,(1,0,0,0),("NONE",)*15)
        return ScenePrediction(
            scene_id=scene.scene_id, raw_relation="PAIR",
            proposed_relation="PAIR", selected_relation="PAIR",
            confidence=.95, consistency=1, accepted=True,
            pair_candidate_id=pair.candidate_id,
            junction_candidate_id=None, identity_log_utility=-.1,
            candidate_count_pair=1, candidate_count_junction=0,
            relation_distribution=(.05,.95,0,0),
            decision_keys_under_perturbation=("PAIR|"+pair.candidate_id,)*15,
        )


def test_train_pilot_with_k10_adapter_route_mask_and_posthoc(monkeypatch,tmp_path):
    original = setup_synthetic_pilot(monkeypatch, tmp_path)
    model = tmp_path / "synthetic_model.joblib"
    model.write_bytes(b"synthetic test only, never joblib loaded")
    monkeypatch.setattr(pilot, "K10_15_SHA256", pilot.digest(model))
    monkeypatch.setattr(ccta_3d_cli.K10Inference,"from_file", lambda path: StubK10())
    output = tmp_path / "k10_pilot"
    report = pilot.execute_train_pilot(
        scan_id=953, output_dir=output, reference_path=original,
        archive_dir=tmp_path, model_pack=model, mode="k10",
        gaps=3, gap_radius_mm=1.1, min_component_voxels=4,
        max_new_voxels=250, repair_radius_mm=0,
    )
    assert report["status"] == "completed_experimental_train_pilot"
    assert report["mode"] == "k10"
    assert (output/"decisions.json").is_file()
    assert (output/"repaired.nii.gz").is_file()
    assert (output/"added_voxels.nii.gz").is_file()
    assert (output/"posthoc_metrics.json").is_file()
    assert "dice_delta" in json.loads((output/"posthoc_metrics.json").read_text())
    assert report["model_pack_sha256"] == pilot.digest(model)
    written = json.loads((output/"decisions.json").read_text())
    assert str(output) in written["output_mask"]
    assert written["input_provenance"]["ct_sha256"] == pilot.digest(output/"ct.nii.gz")


def test_pilot_blocks_nontrain_and_keeps_failed_run_evidence(monkeypatch,tmp_path):
    original = setup_synthetic_pilot(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match="train-only"):
        pilot.execute_train_pilot(
            scan_id=954, output_dir=tmp_path/"blocked",
            reference_path=original, model_pack=None, mode="candidates",
        )
    assert not (tmp_path/"blocked").exists()
    target=tmp_path/"failed_pilot"
    with pytest.raises(ValueError):
        pilot.execute_train_pilot(
            scan_id=953,output_dir=target,reference_path=original,
            archive_dir=tmp_path,model_pack=None,mode="candidates",
            gaps=999,gap_radius_mm=1.1,
        )
    failed=json.loads((target/"pilot_failure.json").read_text())
    assert failed["status"]=="failed_experimental_train_pilot"
    assert failed["error_type"]=="ValueError"
