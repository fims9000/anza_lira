"""Integration checks for opt-in training; synthetic data only."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/ccta_graph_lira_safe_repair"))
from ccta_3d_an_oof import fit_score
import ct28_runtime_window_training as domain
from ccta_3d_endpoints import Endpoint,Candidate,Scene


@pytest.mark.parametrize("variant",["conv3d_sep_structure","anza3d_sep_structure"])
def test_endpoint_jitter_through_real_training(variant):
    torch.set_num_threads(1)
    grid=np.indices((8,8,8),dtype=np.float32)
    x=np.zeros((8,4,8,8,8),dtype=np.float32)
    x[:,0]=np.random.default_rng(7).uniform(-.8,.8,size=x[:,0].shape)
    for i,p in enumerate(([2.,3.,3.],[5.,3.,3.]),1):
        x[:,i]=np.exp(-np.sum((grid-np.array(p)[:,None,None,None])**2,axis=0)/4.5)
    vessel=np.zeros((8,1,8,8,8),dtype=np.float32)
    vessel[:,:,2:6,3,3]=1
    labels=np.tile(np.array([1.,0.],dtype=np.float32),4)
    saved=x.copy()
    _,scores,history=fit_score(x,labels,x.copy(),variant=variant,epochs=1,batch_size=2,
         learning_rate=.001,seed=7,device="cpu",voxel_mm=1.,candidate_type="PAIR",
         train_vessel_targets=vessel,train_scene_ids=np.repeat(["a","b","c","d"],2),
         ranking_weight=.5,endpoint_jitter_mm=.8)
    assert history["endpoint_jitter_seen"]==8
    assert history["endpoint_jitter_applied"]>0
    assert np.isfinite(scores).all() and len(scores)==8 and np.array_equal(x,saved)


def test_runtime_domain_has_patient_oof_rows_and_equal_weight(monkeypatch):
    reference=np.zeros((30,16,16),dtype=bool)
    reference[2:24,5,5]=True
    reference[3:11,10,5]=True
    broken=reference.copy()
    broken[11:14,5,5]=False
    gaps={"gap_radius_mm":1.5,"centers":[{"center_voxel":[12,5,5],"removed_voxels":3}]}
    ends=[Endpoint("a",1,(10,5,5),(-10.,-5.,5.),(1.,0.,0.)),
          Endpoint("b",2,(14,5,5),(-14.,-5.,5.),(-1.,0.,0.)),
          Endpoint("c",3,(10,10,5),(-10.,-10.,5.),(1.,0.,0.))]
    def scenes(ends,patient_label,**kw):
        return [Scene(patient_label+":p",("a","b","c"),
                     (Candidate("PAIR:a+b","PAIR",("a","b")),Candidate("PAIR:a+c","PAIR",("a","c")))),
                Scene(patient_label+":n",("a","c"),(Candidate("PAIR:a+c","PAIR",("a","c")),))]
    loaded=[]
    def load_ct(patient,archive):
        assert patient in domain.TRAIN_CT28_IDS
        loaded.append(patient)
        return np.zeros(reference.shape,dtype=np.float32),np.eye(4)
    monkeypatch.setattr(domain,"load_verified_training_or_validation_ct",load_ct)
    monkeypatch.setattr(domain,"load_aligned_train_reference",lambda *a:(reference,"ref_sha"))
    monkeypatch.setattr(domain,"synthesize_train_gaps",lambda *a,**kw:(broken,gaps))
    monkeypatch.setattr(domain,"extract_endpoints",lambda *a,**kw:(ends,{}))
    monkeypatch.setattr(domain,"build_candidate_scenes",scenes)
    monkeypatch.setattr(domain.graph,"perturbed_candidate_features",lambda c,*a:
                        {k:c[k] for k in ("scene_id","candidate_id","candidate_type")})
    def local(features,fold):
        assert fold in domain.TRAIN_CT28_IDS
        return features.assign(geometry_score=.4,ct_score=.5,geometry_plus_ct_score=.6)
    monkeypatch.setattr(domain.K10Inference,"_score_sparse_candidate_types",staticmethod(local))
    def transform(patient,repeat,ep,cs,ct,inv,scores):
        assert patient in domain.TRAIN_CT28_IDS and len(scores)==len(cs)
        return scores.assign(ct_score=.7)
    _,columns=domain.graph.relation_features.build_historical_features({}, {}, {}, ["geometry_score","ct_score"])
    rows,audit=domain.build_runtime_train_rows(geometry_folds={p:p for p in domain.TRAIN_CT28_IDS},
                 transform=transform,archive_dir=Path("."),columns=columns)
    assert set(loaded)==domain.TRAIN_CT28_IDS
    assert audit["positive_patients"]==17 and audit["positive_windows"]==17 and audit["negative_windows"]==17
    assert len(rows)==17*2*16 and sum(r["sample_weight"] for r in rows)==pytest.approx(1036.)
    assert len(columns)==84 and {r["relation_class"] for r in rows}=={"PAIR","NONE"}
    assert all(r["split"]=="train" and r["scan_id"] in domain.TRAIN_CT28_IDS for r in rows)
