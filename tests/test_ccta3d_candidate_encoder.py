"""Identical volumetric CT patch interface for ANZA/CNN controlled ablation."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.ccta3d_candidate_encoder import Candidate3dEncoder,parameter_budget


def test_equal_patch_inputs_and_matched_parameter_budget():
    torch.manual_seed(11)
    az=Candidate3dEncoder("anza3d")
    cnn=Candidate3dEncoder("conv3d")
    budget=parameter_budget(az,cnn)
    assert budget["matched_parameter_budget"],budget
    assert budget["relative_gap"]<.20
    patch=torch.randn(2,2,10,10,10)
    a=az(patch)
    b=cnn(patch)
    assert a.shape==b.shape==(2,)
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()


@pytest.mark.parametrize("architecture",["anza3d","conv3d"])
def test_candidate_head_gradients_are_finite(architecture):
    torch.manual_seed(17)
    model=Candidate3dEncoder(architecture,channels=9)
    patches=torch.randn(1,2,8,8,8,requires_grad=True)
    target=torch.tensor([1.])
    logits=model(patches)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    loss.backward()
    assert torch.isfinite(patches.grad).all()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()


def test_reject_wrong_patch_shape_or_variant():
    with pytest.raises(ValueError):
        Candidate3dEncoder("unknown")
    with pytest.raises(ValueError,match="2-channel"):
        Candidate3dEncoder("conv3d")(torch.randn(2,1,8,8,8))
