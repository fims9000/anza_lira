"""End-to-end 15-repeat neural CT adapter with synthetic model stubs only."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import run_ct28_graph_lira as graph
from ccta_3d_endpoints import Candidate,Endpoint,Scene
from ccta_3d_an_patches import PatchConfig
from ccta_3d_an_runtime import NeuralCCTA3dInference
from ccta_3d_an_cli import run_neural
from build_ct28_relation_features import HISTORICAL_FEATURES_PER_TYPE,CROSS_FEATURES


class Binary:
    def __init__(self,columns):
        self.feature_names_in_=np.array(columns)
    def predict_proba(self,values):
        return np.tile([.03,.97],(len(values),1))


class Relation:
    classes_=np.array([0,1,2,3])
    def predict_proba(self,values):
        return np.tile([.01,.97,.01,.01],(len(values),1))


class FixedNeural(torch.nn.Module):
    def forward(self,x):
        return torch.zeros(len(x),device=x.device)


def model_columns(streams):
    result=[]
    for stream in streams:
        for kind in ("PAIR","JUNCTION"):
            result.extend(
                f"{stream}_{kind.lower()}_{feature}"
                for feature in HISTORICAL_FEATURES_PER_TYPE
            )
        result.extend(f"{stream}_{feature}" for feature in CROSS_FEATURES)
    return result


def models():
    return {
        "local_models":{
            "PAIR":{
                "geometry":Binary(["pair_g0"]),
                "ct":Binary(["pair_ct0"]),
                "geometry_plus_ct":Binary(["pair_g0","pair_ct0"]),
            },
            "JUNCTION":{
                "geometry":Binary(["junction_span_mm"]),
                "ct":Binary(["junction_ct0"]),
                "geometry_plus_ct":Binary(["junction_span_mm","junction_ct0"]),
            },
        },
        "relation_heads":{
            "geometry_only":Relation(),
            "ct_conditioned":Relation(),
        },
        "geometry_relation_features":model_columns(("geometry_score",)),
        "ct_conditioned_relation_features":model_columns(
            ("geometry_score","ct_score")
        ),
    }


def setup():
    eps=[
        Endpoint(f"E{i:04d}",i+1,(11+i,15,15),
                 (-(11.+i),-15.,15.),(1.,0.,0.))
        for i in range(5)
    ]
    scene=Scene(
        "REAL3D:train_synthetic:W0000",
        tuple(e.endpoint_id for e in eps),
        (
            Candidate("PAIR:E0000+E0001","PAIR",("E0000","E0001")),
            Candidate("JUNCTION:E0002+E0003+E0004","JUNCTION",
                      ("E0002","E0003","E0004")),
        ),
    )
    return eps,scene


def test_real_3d_patch_neural_scores_replace_only_ct_relation_evidence(monkeypatch):
    def dummy_features(row,lookup,ct,inverse,arm_names):
        return {
            "scene_id":row.scene_id,
            "candidate_id":row.candidate_id,
            "candidate_type":row.candidate_type,
            "pair_g0":1.,"pair_ct0":100.,
            "junction_span_mm":2.,"junction_ct0":100.,
        }
    monkeypatch.setattr(graph,"perturbed_candidate_features",dummy_features)
    endpoint_list,scene=setup()
    engine=NeuralCCTA3dInference(
        models(),pair_encoder=FixedNeural(),
        junction_encoder=FixedNeural(),
        patch_config=PatchConfig(side=8),
        variant="anza3d",
    )
    _,ends,cands=engine._frames(scene,endpoint_list)
    scores,decisions=engine._score(
        scene,ends,cands,np.full((32,32,32),250.,dtype=np.float32),
        np.eye(4)
    )
    assert np.allclose(scores["ct_score"].to_numpy(),.5)
    assert np.allclose(scores["geometry_score"].to_numpy(),.97)
    prediction=engine.predict(
        scene,endpoint_list,np.full((32,32,32),250.,dtype=np.float32),
        np.eye(4)
    )
    assert prediction.accepted
    assert prediction.proposed_relation=="PAIR"
    assert prediction.consistency==1.
    assert len(prediction.decision_keys_under_perturbation)==15


def test_neural_cli_forbids_heldout_before_loading_medical_files():
    dummy=Path("/nonexistent/medical/ct.nii.gz")
    with pytest.raises(ValueError,match="held-out"):
        run_neural(
            ct=dummy,mask=dummy,patient_id="954",
            base_k10=dummy,relation_dir=dummy,
            pair_oof=dummy,junction_oof=dummy,variant="anza3d",
            output_mask=dummy,report=dummy,
        )


def test_neural_encoder_persists_full_volume_nifti_and_provenance(monkeypatch,tmp_path):
    import json
    import nibabel as nib
    from ccta_3d_cli import run_cli
    def stub_features(row,lookup,ct,inverse,arm_names):
        return {
            "scene_id":row.scene_id,
            "candidate_id":row.candidate_id,
            "candidate_type":row.candidate_type,
            "pair_g0":2.0,
            "pair_ct0":240.0,
            "junction_span_mm":3.0,
            "junction_ct0":230.0,
        }
    monkeypatch.setattr(graph,"perturbed_candidate_features",stub_features)
    ct=np.full((50,65,35),250.,dtype=np.float32)
    mask=np.zeros(ct.shape,dtype=np.uint8)
    for y in (10,16,28,40,52):
        mask[10:19,y,18]=1
    ctfile=tmp_path/"ct.nii.gz"
    maskfile=tmp_path/"broken.nii.gz"
    nib.save(nib.Nifti1Image(ct,np.eye(4)),ctfile)
    nib.save(nib.Nifti1Image(mask,np.eye(4)),maskfile)
    engine=NeuralCCTA3dInference(
        models(),pair_encoder=FixedNeural(),junction_encoder=FixedNeural(),
        patch_config=PatchConfig(side=8),variant="anza3d",
    )
    outputs={
        "ct_path":ctfile,
        "mask_path":maskfile,
        "patient_id":"train_synthetic_anza",
        "mode":"k10",
        "model_pack":tmp_path/"stub_not_read.joblib",
        "output_mask":tmp_path/"repaired.nii.gz",
        "output_additions":tmp_path/"added.nii.gz",
        "report_path":tmp_path/"decision.json",
        "radius_mm":0,
        "max_new_voxels":50,
        "min_component_voxels":5,
        "trace_mm":3,
        "max_endpoints":30,
        "max_scenes":128,
        "threshold":.5,
        "engine_override":engine,
        "experimental_evidence":{
            "variant":"anza3d",
            "source":"synthetic_stub_no_medical_CT",
        }
    }
    outcome=run_cli(**outputs)
    persisted=json.loads(outputs["report_path"].read_text())
    assert persisted["experimental_evidence"]["variant"]=="anza3d"
    assert outcome["experimental_encoder_variant"]=="anza3d"
    assert outcome["model_pack_sha256"]=="synthetic_test_only"
    assert persisted["scene_count"]>=1
    assert persisted["accepted_decisions"]>=0
    repaired=nib.load(str(outputs["output_mask"]))
    added=nib.load(str(outputs["output_additions"]))
    assert repaired.shape==ct.shape and added.shape==ct.shape
    before=np.asanyarray(nib.load(str(maskfile)).dataobj)>0
    after=np.asanyarray(repaired.dataobj)>0
    increase=np.asanyarray(added.dataobj)>0
    assert np.array_equal(increase,after&~before)
    assert np.all(before<=after)


def test_true_volumetric_anisotropic_anza_model_scores_real_3d_ct_patches(monkeypatch):
    from models.ccta3d_candidate_encoder import Candidate3dEncoder
    def dummy_features(row,lookup,ct,inverse,arm_names):
        return {
            "scene_id":row.scene_id,
            "candidate_id":row.candidate_id,
            "candidate_type":row.candidate_type,
            "pair_g0":1.0,
            "pair_ct0":200.0,
            "junction_span_mm":2.0,
            "junction_ct0":230.0,
        }
    monkeypatch.setattr(graph,"perturbed_candidate_features",dummy_features)
    ends,scene=setup()
    torch.manual_seed(45)
    neural=NeuralCCTA3dInference(
        models(),
        pair_encoder=Candidate3dEncoder("anza3d",channels=9,voxel_mm=1.25),
        junction_encoder=Candidate3dEncoder("anza3d",channels=9,voxel_mm=1.25),
        patch_config=PatchConfig(side=8,voxel_mm=1.25),
        variant="anza3d",
    )
    _,endpoint_frame,candidate_frame=neural._frames(scene,ends)
    ct=np.random.default_rng(21).normal(
        250.,50.,size=(32,32,32)
    ).astype(np.float32)
    scores,decisions=neural._score(
        scene,endpoint_frame,candidate_frame,ct,np.eye(4)
    )
    assert len(scores)==2
    assert scores["ct_score"].between(0,1).all()
    assert np.isfinite(scores["ct_score"].to_numpy()).all()
    assert not np.allclose(scores["ct_score"].to_numpy(),.5)
    assert "ct_conditioned" in decisions
