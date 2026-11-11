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


@pytest.mark.parametrize("variant",["conv3d_structure","anza3d_structure"])
def test_structural_fit_requires_train_only_dense_targets_and_produces_scores(variant):
    data,labels,targets=_sample()
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
