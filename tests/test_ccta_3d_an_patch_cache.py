"""Synthetic CT28 CSV fixture tests for 3D patch cache, with no medical data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_patch_cache import build_patient_cache
from ccta_3d_an_patches import PatchConfig


def fake_scene_csvs(tmp_path):
    scene=tmp_path/"scene"
    scene.mkdir()
    pd.DataFrame([{
        "scene_id":"train_953_scene_a","scan_id":953,"split":"train"
    }]).to_csv(scene/"scene_plan_train_val.csv",index=False)
    ep=[]
    for i in range(5):
        ep.append({
            "scene_id":"train_953_scene_a",
            "endpoint_id":f"E{i:02d}",
            "x_lps":-15.0-i,
            "y_lps":-14.0,
            "z_lps":14.0,
        })
    pd.DataFrame(ep).to_csv(scene/"scene_endpoints_train_val.csv",index=False)
    pd.DataFrame([
        {"scene_id":"train_953_scene_a",
         "candidate_id":"PAIR:E00+E01",
         "candidate_type":"PAIR","endpoint_ids":"E00|E01","y":1},
        {"scene_id":"train_953_scene_a",
         "candidate_id":"JUNCTION:E01+E02+E03",
         "candidate_type":"JUNCTION","endpoint_ids":"E01|E02|E03","y":0},
    ]).to_csv(scene/"scene_candidates_train_val.csv",index=False)
    return scene


def test_train_candidate_patch_cache_is_patient_isolated(tmp_path):
    scene=fake_scene_csvs(tmp_path)
    output=tmp_path/"out"
    def ct_provider(patient):
        assert patient==953
        return np.full((40,40,40),260,dtype=np.float32),np.eye(4)
    receipt=build_patient_cache(
        patient=953,scenes_dir=scene,archive=tmp_path,
        output_dir=output,config=PatchConfig(side=16),
        load_volume=ct_provider,
    )
    assert receipt["patient_id"]==953
    assert receipt["n_candidates"]==2
    assert receipt["n_positives"]==1
    assert receipt["held_out_test_accessed"] is False
    assert receipt["model_input_contains_ground_truth"] is False
    data=np.load(output/"patches_953.npz",allow_pickle=False)
    assert data["patches"].shape==(2,2,16,16,16)
    assert list(data["y"])==[1,0]
    assert set(data["candidate_type"])=={"PAIR","JUNCTION"}
    assert np.isfinite(data["patches"]).all()
    with pytest.raises(FileExistsError):
        build_patient_cache(
            patient=953,scenes_dir=scene,output_dir=output,
            config=PatchConfig(side=16),load_volume=ct_provider,
        )


def test_patch_cache_never_opens_heldout_test_or_validation_by_default(tmp_path):
    for pid in (954,957,980):
        with pytest.raises(ValueError,match="train"):
            build_patient_cache(
                patient=pid,scenes_dir=tmp_path,
                output_dir=tmp_path/"out",
                load_volume=lambda _: (_ for _ in ()).throw(
                    AssertionError("forbidden CT volume opened")
                )
            )


def test_dense_vessel_supervision_never_enters_two_channel_ct_input(tmp_path,monkeypatch):
    """Both candidate scorers see byte-identical inputs with/without target."""
    import ccta_3d_an_patch_cache as cache
    scene=fake_scene_csvs(tmp_path)
    def ct_provider(patient):
        assert patient==953
        return np.full((40,40,40),260,dtype=np.float32),np.eye(4)
    normal=tmp_path/"normal"
    supervised=tmp_path/"supervised"
    mask=np.zeros((40,40,40),dtype=np.uint8)
    mask[13:23,14,14]=1
    def reference_provider(patient,shape,affine,segmentation_dir,train_ids):
        assert patient==953 and patient in train_ids
        assert tuple(shape)==(40,40,40)
        assert np.allclose(affine,np.eye(4))
        return mask,"test-only-segmentation-sha"
    monkeypatch.setattr(cache,"load_aligned_train_reference",reference_provider)
    base=build_patient_cache(
        patient=953,scenes_dir=scene,output_dir=normal,
        config=PatchConfig(side=16,frame_mode="fixed_lps"),load_volume=ct_provider,
    )
    rich=build_patient_cache(
        patient=953,scenes_dir=scene,output_dir=supervised,
        config=PatchConfig(side=16,frame_mode="fixed_lps"),load_volume=ct_provider,
        with_vessel_targets=True,
    )
    with np.load(normal/"patches_953.npz",allow_pickle=False) as baseline, \
         np.load(supervised/"patches_953.npz",allow_pickle=False) as structure:
        assert set(structure.files)-set(baseline.files)=={"vessel_targets"}
        assert np.array_equal(baseline["patches"],structure["patches"])
        assert np.array_equal(baseline["y"],structure["y"])
        assert structure["vessel_targets"].shape==(2,16,16,16)
        assert (structure["vessel_targets"].sum(axis=(1,2,3))>0).all()
        assert structure["vessel_targets"].dtype==np.uint8
    assert base["train_only_dense_supervision"] is False
    assert rich["train_only_dense_supervision"] is True
    assert rich["reference_segmentation_sha256"]=="test-only-segmentation-sha"
    assert rich["model_input_contains_ground_truth"] is False
