"""Falsifiable structural readout/anisotropy checks (no medical CT in CI)."""
import math
import torch
import pytest

from models.azconv3d import AZConv3dConfig, quadratic_metric, pair_kernel
from models.ccta3d_candidate_encoder import Candidate3dEncoder, parameter_budget
from models.structural_bridge3d import VesselBridgeReadout, vessel_structure_loss


def _heat(endpoints,side=12):
    xyz=torch.stack(torch.meshgrid(
        *(torch.arange(side,dtype=torch.float32) for _ in range(3)),
        indexing="ij",
    ),dim=0)
    heat=torch.zeros((side,side,side))
    for point in endpoints:
        pos=torch.tensor(point,dtype=torch.float32)[:,None,None,None]
        heat+=torch.exp(-((xyz-pos)**2).sum(dim=0)/(2*1.5**2))
    return heat.clamp(0,1)[None]


def _tube(paths,side=12,missing_arm=None):
    xyz=torch.stack(torch.meshgrid(
        *(torch.arange(side,dtype=torch.float32) for _ in range(3)),
        indexing="ij",
    ),dim=0)
    support=torch.zeros(side,side,side)
    for j,(a,b) in enumerate(paths):
        if j==missing_arm:
            continue
        a=torch.tensor(a,dtype=torch.float32)
        b=torch.tensor(b,dtype=torch.float32)
        for t in torch.linspace(0,1,80):
            pos=(a+(b-a)*t)[:,None,None,None]
            support=torch.maximum(support,torch.exp(-((xyz-pos)**2).sum(0)/1.0))
    return (0.05+0.9*support)[None]


def test_negative_h_prefers_axial_transport_in_real_physical_metric():
    axis=torch.tensor([[[[[[1.]]],[[[0.]]],[[[0.]]]]]],dtype=torch.float64)
    assert axis.shape==(1,1,3,1,1,1)
    hyper=torch.full((1,1,1,1,1),-.8,dtype=torch.float64)
    along=quadratic_metric(axis,hyper,torch.tensor([1.,0.,0.],dtype=torch.float64))
    across=quadratic_metric(axis,hyper,torch.tensor([0.,1.,0.],dtype=torch.float64))
    assert along.item()<across.item()
    ones=torch.ones_like(hyper)
    first=pair_kernel(ones,ones,axis,axis,hyper,hyper,
                      torch.tensor([1.,0.,0.],dtype=torch.float64),beta=.12)
    second=pair_kernel(ones,ones,axis,axis,hyper,hyper,
                       torch.tensor([0.,1.,0.],dtype=torch.float64),beta=.12)
    assert first.item()>second.item()
    assert AZConv3dConfig(axial_preference=True).axial_preference


def test_pair_connectivity_decreases_for_missing_real_path():
    ends=[[2.,6.,6.],[9.,6.,6.]]
    support=_tube([(ends[0],ends[1])])
    interrupted=support.clone()
    interrupted[:,5:7,:,:]=0.05
    heat=_heat(ends)
    op=VesselBridgeReadout(2,curvature=1.6)
    good=op(support,heat)
    bad=op(interrupted,heat)
    assert good.item()>bad.item()+.25,(good.item(),bad.item())
    assert 0<bad.item()<1


def test_junction_requires_three_arms_not_only_two():
    ends=[[2.,6.,6.],[9.,6.,6.],[6.,9.,6.]]
    hub=[sum(x[i] for x in ends)/3 for i in range(3)]
    paths=[(x,hub) for x in ends]
    heat=_heat(ends)
    readout=VesselBridgeReadout(3)
    full=readout(_tube(paths),heat)
    missing=readout(_tube(paths,missing_arm=2),heat)
    assert full.item()>missing.item()+.08,(full.item(),missing.item())


@pytest.mark.parametrize("name",["conv3d_structure","anza3d_structure"])
def test_model_uses_dense_structure_and_backprop_on_volumetric_input(name):
    torch.manual_seed(41)
    model=Candidate3dEncoder(name,channels=6,candidate_type="PAIR")
    original=torch.randn(2,2,10,10,10)
    original[:,1]=_heat([[2.,5.,5.],[7.,5.,5.]],side=10)
    label=torch.tensor([1.,0.])
    truth=torch.zeros(2,1,10,10,10)
    truth[:,:,3:7,5,5]=1
    logit,dense,bridge=model(original,return_structure=True)
    assert logit.shape==(2,) and dense.shape==(2,1,10,10,10)
    assert bridge.shape==(2,) and ((bridge>0)&(bridge<1)).all()
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,label)
          + .25*vessel_structure_loss(dense,truth,skeleton_iterations=1))
    loss.backward()
    for module in (model.stem,model.vessel_head,model.local):
        grad=[p.grad for p in module.parameters() if p.requires_grad]
        assert grad and all(g is not None and torch.isfinite(g).all() for g in grad)
    assert model.bridge_strength_logit.grad is not None
    assert torch.isfinite(model.bridge_strength_logit.grad)


def test_structural_variants_have_comparable_capacity_and_are_opt_in():
    pair=Candidate3dEncoder("anza3d_structure")
    ref=Candidate3dEncoder("conv3d_structure")
    ratio=parameter_budget(pair,ref)
    assert ratio["matched_parameter_budget"],ratio
    assert pair.local.cfg.axial_preference
    noax=Candidate3dEncoder("anza3d_noaxial_structure")
    assert noax.structure_aware
    assert not noax.local.cfg.axial_preference
    assert noax.parameter_count()==pair.parameter_count()
    assert not Candidate3dEncoder("anza3d").local.cfg.axial_preference
    with pytest.raises(ValueError,match="not enabled"):
        Candidate3dEncoder("conv3d")(torch.randn(1,2,8,8,8),return_structure=True)


def test_soft_cldice_has_finite_value_on_sparse_and_empty_vessel_targets():
    torch.manual_seed(19)
    pred=torch.randn(2,1,8,8,8,requires_grad=True)
    truth=torch.zeros_like(pred)
    truth[1,0,2:7,4,4]=1
    loss=vessel_structure_loss(pred,truth,skeleton_iterations=2)
    assert loss.item()>0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_exact_3d_reachability_separates_connected_and_disconnected_junction():
    from models.structural_bridge3d import FuzzyConnectedness3d
    heat=_heat([[2.,6.,6.],[9.,6.,6.],[6.,9.,6.]])
    mask=torch.zeros((1,12,12,12))
    mask[:,2:10,6,6]=1
    mask[:,6,6:10,6]=1
    strong=FuzzyConnectedness3d(3,steps=24)(mask,heat)
    broken=mask.clone()
    broken[:,6,7:9,6]=0
    weak=FuzzyConnectedness3d(3,steps=24)(broken,heat)
    assert strong.item()>.99
    assert weak.item()<.01


def test_fuzzy_reachability_propagates_finite_gradients():
    from models.structural_bridge3d import FuzzyConnectedness3d
    heat=_heat([[2.,6.,6.],[9.,6.,6.],[6.,9.,6.]])
    vessel=torch.randn(1,12,12,12,requires_grad=True)
    probability=vessel.sigmoid()
    score=FuzzyConnectedness3d(3,steps=24)(probability,heat)
    score.sum().backward()
    assert torch.isfinite(vessel.grad).all()
    assert vessel.grad.abs().sum().item()>0


def test_adaptive_three_arm_hub_recovers_offset_junction_but_rejects_missing_arm():
    """True branch hub displaced from the endpoint centroid by two voxels."""
    ends=[[2.,6.,6.],[9.,6.,6.],[6.,9.,6.]]
    centroid=torch.tensor(ends).mean(dim=0).tolist()
    hub=[centroid[0],centroid[1]-2.,centroid[2]]
    arms=[(p,hub) for p in ends]
    heat=_heat(ends)
    vessel=_tube(arms)
    old=VesselBridgeReadout(3,junction_strategy="centroid_star")(vessel,heat)
    adaptive=VesselBridgeReadout(3,junction_strategy="adaptive_star")(vessel,heat)
    rejected=VesselBridgeReadout(3,junction_strategy="adaptive_star")(
        _tube(arms,missing_arm=2),heat
    )
    assert adaptive.item()>old.item()+.05,(old.item(),adaptive.item())
    assert adaptive.item()>rejected.item()+.10,(adaptive.item(),rejected.item())


def test_adaptive_hub_has_finite_gradients_and_equal_cnn_anza_contract():
    vessel=torch.randn(1,12,12,12,requires_grad=True).sigmoid()
    heat=_heat([[2.,6.,6.],[9.,6.,6.],[6.,9.,6.]])
    out=VesselBridgeReadout(3,junction_strategy="adaptive_star")(vessel,heat)
    grad=torch.autograd.grad(out.sum(),vessel,allow_unused=False)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    for variant in ("conv3d_structure","anza3d_structure","anza3d_noaxial_structure"):
        net=Candidate3dEncoder(variant,candidate_type="JUNCTION",channels=6)
        assert net.bridge_readout.junction_strategy=="adaptive_star"
    with pytest.raises(ValueError,match="strategy"):
        VesselBridgeReadout(3,junction_strategy="nonexistent")


def test_connected_tree_does_not_imply_same_branch_identity():
    """Adversarial PAIR case: two candidate paths share one connected tree.

    The false pair crosses a Y bifurcation from one anatomical branch onto
    another. A connectivity-only oracle cannot determine whether the repair
    should join these endpoints; a separate branch identity contract is needed.
    """
    from models.structural_bridge3d import FuzzyConnectedness3d
    from scipy.ndimage import label as cc_label
    import numpy as np

    mask=torch.zeros((1,12,12,12))
    mask[:,2:10,6,6]=1.
    mask[:,6,6:10,6]=1.
    good_heat=_heat([[2.,6.,6.],[9.,6.,6.]])
    wrong_heat=_heat([[2.,6.,6.],[6.,9.,6.]])
    comp,_=cc_label(mask[0].numpy()>0,np.ones((3,3,3),dtype=np.uint8))
    assert comp[2,6,6]==comp[9,6,6]==comp[6,9,6]>0
    connected=FuzzyConnectedness3d(degree=2,steps=24)
    a=connected(mask,good_heat)
    b=connected(mask,wrong_heat)
    assert a.item()>.99 and b.item()>.99
    # "Wrong" here is a branch-identity decision, *not* an assertion that
    # its endpoints are in different physical components.
