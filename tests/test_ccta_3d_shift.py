"""Train-only CT28 cardinality support and strict OOD abstention tests."""
from pathlib import Path
import sys
from collections import Counter

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/
                      "scripts/research/ccta_graph_lira_safe_repair"))
import ccta_3d_runtime as runtime
from ccta_3d_shift import training_count_support,describe_shift,scene_counts
from ccta_3d_endpoints import Scene,Candidate
from ct28_path_stage import PathConfig


def test_frozen_train_counts_are_loaded_without_val_labels():
    support=training_count_support()
    assert sum(support.values())==1036
    assert len(support)>0
    assert min(key[0] for key in support)>=0


def test_unsupported_candidate_counts_are_explicit():
    scene=Scene(
        "real3d:demo",( "e1","e2","e3","e4","e5"),(
            Candidate("PAIR:e1+e2","PAIR",("e1","e2")),
        ),
    )
    audit=describe_shift([scene],Counter({(7,3):11}))
    assert audit["runtime_scenes"]==1
    assert audit["out_of_train_count_support"]==1
    assert audit["unsupported_scenes"][0]["pair_candidates"]==1
    assert scene_counts(scene)==(1,0)


def test_strict_ood_policy_rejects_model_acceptance(monkeypatch):
    class AcceptAll:
        pack_sha="synthetic"
        def predict(self,scene,endpoints,ct,affine):
            from ccta_3d_k10 import ScenePrediction
            pair=next(c for c in scene.candidates if c.kind=="PAIR")
            return ScenePrediction(
                scene.scene_id,"PAIR","PAIR","PAIR",.98,1.,True,
                pair.candidate_id,None,-.1,1,0,
                (.01,.98,.01,0.),("PAIR|"+pair.candidate_id,)*15
            )
    from tests.test_ccta_3d_runtime import synthetic_case
    ct,mask,affine=synthetic_case()
    monkeypatch.setattr(runtime,"training_count_support",lambda:Counter({(99,99):1036}))
    out,report=runtime.run_volume(
        ct,mask,affine,patient_label="synthetic_no_such_count",
        engine=AcceptAll(),
        params=runtime.RuntimeConfig(radius_mm=0,max_new_voxels=100,
                                     min_component_voxels=5),
        path_config=PathConfig(),
        strict_count_support=True,
    )
    assert np.array_equal(out,mask)
    assert report["accepted_decisions"]==0
    assert report["candidate_count_shift"]["out_of_train_count_support"]>0
    assert report["global_selection"]["eligible_K10_proposals"]==0
    assert all(decision["reason"]=="invalid_global_proposal:out_of_training_count_support"
               for decision in report["decisions"])
