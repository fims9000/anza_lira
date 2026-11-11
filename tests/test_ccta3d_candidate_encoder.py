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


def test_anisotropic_kernel_uses_patch_voxel_mm_at_inference():
    torch.manual_seed(73)
    coarse=Candidate3dEncoder("anza3d",channels=9,voxel_mm=2.5)
    fine=Candidate3dEncoder("anza3d",channels=9,voxel_mm=.5)
    fine.load_state_dict(coarse.state_dict())
    sample=torch.randn(1,2,5,6,5)
    out_coarse=coarse(sample)
    out_fine=fine(sample)
    assert torch.isfinite(out_coarse).all()
    assert torch.isfinite(out_fine).all()
    assert not torch.allclose(out_coarse,out_fine,rtol=1e-7,atol=1e-7)
    with pytest.raises(ValueError,match="Physical patch voxel"):
        Candidate3dEncoder("anza3d",voxel_mm=0)


def test_contrastive_anza_candidate_scores_and_gradients_share_input_contract():
    torch.manual_seed(20261112)
    baseline=Candidate3dEncoder(
        "anza3d_sep_structure",channels=6,candidate_type="PAIR"
    )
    contrast=Candidate3dEncoder(
        "anza3d_contrast_sep_structure",channels=6,candidate_type="PAIR"
    )
    contrast.load_state_dict(baseline.state_dict(),strict=True)
    assert contrast.local.cfg.contrastive
    assert not baseline.local.cfg.contrastive
    assert contrast.local.cfg.axial_preference==baseline.local.cfg.axial_preference
    assert contrast.parameter_count()==baseline.parameter_count()
    x=torch.randn(2,4,8,8,8,requires_grad=True)
    x.data[:,3]=0
    logits,vessel,bridge=contrast(x,return_structure=True)
    assert logits.shape==(2,) and vessel.shape==(2,1,8,8,8)
    assert bridge.shape==(2,)
    assert torch.isfinite(logits).all()
    loss=logits.square().mean()+.05*vessel.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert contrast.local.values.weight.grad is not None
    assert contrast.local.values.weight.grad.abs().sum()>0
