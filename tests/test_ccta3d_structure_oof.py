"""Patient-split and dense-mask supervision safety checks for structure variants."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/research/ccta_graph_lira_safe_repair"))

from ccta_3d_an_oof import fit_score
from ccta_3d_structure_targets import sample_vessel_reference,load_aligned_train_reference
from ccta_3d_an_patches import PatchConfig


def _sample(n=4,side=8):
    rng=np.random.default_rng(7)
    x=np.zeros((n,2,side,side,side),dtype=np.float32)
    x[:,0]=rng.uniform(-.8,.8,size=(n,side,side,side))
    z,y,w=np.meshgrid(*(np.arange(side) for _ in range(3)),indexing="ij")
    for row in range(n):
        x[row,1]=(
            np.exp(-((z-1)**2+(y-4)**2+(w-4)**2)/(2*1.5**2))+
            np.exp(-((z-6)**2+(y-4)**2+(w-4)**2)/(2*1.5**2))
        ).clip(0,1)
    truth=np.zeros((n,1,side,side,side),dtype=np.float32)
    truth[:,:,2:6,4,4]=1
    y=np.asarray([1.,0.,1.,0.],dtype=np.float32)
    return x,y,truth



def _maybe_separate(x:np.ndarray,kind:str,variant:str)->np.ndarray:
    if "_sep_" not in variant:
        return x
    b,_,side,_,_=x.shape
    xyz=np.stack(np.meshgrid(*(np.arange(side) for _ in range(3)),indexing="ij"))
    pts=[[1.,side//2,side//2],[side-2.,side//2,side//2]]
    if kind=="JUNCTION":
        pts.append([side//2,2.,side//2])
    out=np.zeros((b,4,side,side,side),dtype=np.float32)
    out[:,0]=x[:,0]
    for i,p in enumerate(pts):
        dist=((xyz-np.asarray(p)[:,None,None,None])**2).sum(axis=0)
        out[:,1+i]=np.exp(-dist/(2*1.5**2))
    return out


@pytest.mark.parametrize("variant",["conv3d_structure","anza3d_structure","conv3d_sep_structure","anza3d_sep_structure"])
def test_structural_fit_requires_train_only_dense_targets_and_produces_scores(variant):
    data,labels,targets=_sample()
    data=_maybe_separate(data,"PAIR",variant)
    with pytest.raises(ValueError,match="train-only vessel"):
        fit_score(data,labels,data,variant=variant,epochs=1,batch_size=2,
                  learning_rate=.001,seed=10,device="cpu",voxel_mm=1.,
                  candidate_type="PAIR")
    network,scores,history=fit_score(
        data,labels,data,variant=variant,epochs=1,batch_size=2,
        learning_rate=.001,seed=10,device="cpu",voxel_mm=1.,
        candidate_type="PAIR",train_vessel_targets=targets,
    )
    assert scores.shape==(4,) and np.isfinite(scores).all()
    assert np.all((0<=scores)&(scores<=1))
    assert history["structure_training"]["dense_loss_weight"]==pytest.approx(.35)
    assert network.structure_aware


def test_baseline_cannot_accidentally_consume_vessel_targets():
    data,labels,targets=_sample()
    with pytest.raises(ValueError,match="baseline"):
        fit_score(data,labels,data,variant="conv3d",epochs=1,batch_size=2,
                  learning_rate=.001,seed=10,device="cpu",voxel_mm=1.,
                  train_vessel_targets=targets)


def test_reference_loading_refuses_validation_before_opening_files(tmp_path):
    with pytest.raises(ValueError,match="train"):
        load_aligned_train_reference(957,(16,16,16),np.eye(4),tmp_path,{953,955})


def test_dense_mask_uses_same_physical_patch_transform_and_binary_dtype():
    mask=np.zeros((30,30,30),dtype=np.uint8)
    mask[14:19,15,15]=1
    # Identity maps voxel indices to RAS. LPS is x/y sign-inverted.
    endpoints=np.array([[-14.,-15.,15.],[-18.,-15.,15.]],dtype=float)
    output=sample_vessel_reference(mask,np.eye(4),endpoints,
                                   PatchConfig(side=12,frame_mode="fixed_lps"))
    assert output.shape==(12,12,12)
    assert output.dtype==np.uint8
    assert output.sum()>0


@pytest.mark.parametrize("variant",["conv3d_structure","anza3d_structure","conv3d_sep_structure","anza3d_sep_structure"])
@pytest.mark.parametrize("kind",["PAIR","JUNCTION"])
def test_complete_patient_oof_structure_model_checkpoint_reload(
    tmp_path,variant,kind
):
    """End-to-end two-fold patient-isolated fit/load without real medical CT."""
    import hashlib
    import json
    from ccta_3d_an_oof import build_oof
    from models.ccta3d_candidate_encoder import Candidate3dEncoder

    patients=[953,955,956]
    val_patient=957
    cache=tmp_path/"cache"
    cache.mkdir()
    base_x,base_y,base_mask=_sample()
    base_x=_maybe_separate(base_x,kind,variant)
    for p in [*patients,val_patient]:
        is_train=p in patients
        npz=cache/f"patches_{p}.npz"
        # Repeated spatial data across patients is a *numerical integration*
        # fixture only; it must never be cited as a patient generalization test.
        x=(base_x+.01*(p%5)).astype(np.float16)
        candidate_ids=[f"{kind}:e{j}" for j in range(4)]
        scene_ids=[f"p{p}_s{j//2}" for j in range(4)]
        arrays={
            "patches":x,
            "candidate_type":np.asarray([kind]*4),
            "y":base_y.astype(np.uint8),
            "candidate_id":np.asarray(candidate_ids),
            "scene_id":np.asarray(scene_ids),
        }
        if is_train:
            arrays["vessel_targets"]=base_mask[:,0].astype(np.uint8)
        np.savez_compressed(npz,**arrays)
        digest=hashlib.sha256(npz.read_bytes()).hexdigest()
        receipt={
            "patient_id":p,
            "split":"train" if is_train else "val",
            "cache_sha256":digest,
            "patch_protocol":{"side":8,"voxel_mm":1.0,"frame_mode":"fixed_lps",
                              "endpoint_encoding":"separate" if "_sep_" in variant else "summed"},
            "model_input_contains_ground_truth":False,
            "endpoint_outside_patch_count":0,
            "input_control_mode":"full_ct",
            "train_only_dense_supervision":is_train,
        }
        (cache/f"patches_{p}.json").write_text(json.dumps(receipt))
    path=tmp_path/"oof"
    report=build_oof(
        cache_dir=cache,out_dir=path,candidate_type=kind,variant=variant,
        train_patients=patients,val_patients=[val_patient],
        folds=2,epochs=1,batch_size=4,seed=11,device="cpu",
        allow_subset=True,
    )
    assert report["fold_count"]==2
    assert report["n_train_patients"]==3
    assert report["n_validation_patients"]==1
    assert report["held_out_test_accessed"] is False
    assert report["structure_objective"]["vessel_supervision"]=="train_only"
    assert sum(len(x["held_patients"]) for x in report["fold_assignments"])==3
    for fold in report["fold_assignments"]:
        assert not set(fold["train_patients"])&set(fold["held_patients"])
    scores=(path/f"{kind}_{variant}_oof_val.csv").read_text()
    assert "oof_fold" in scores and "val" in scores
    checkpoint=path/f"{kind}_{variant}_fulltrain.pt"
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    loaded=Candidate3dEncoder(variant,candidate_type=kind)
    loaded.load_state_dict(state,strict=True)
    loaded.eval()
    with torch.no_grad():
        pred=loaded(torch.from_numpy(base_x[:1]))
    assert pred.shape==(1,) and torch.isfinite(pred).all()
