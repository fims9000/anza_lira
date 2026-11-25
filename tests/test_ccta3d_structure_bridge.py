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


@pytest.mark.parametrize("variant",[
    "conv3d_sep_structure","anza3d_sep_structure","anza3d_noaxial_sep_structure"
])
def test_separate_tip_model_is_permutation_symmetric_and_has_finite_gradients(variant):
    torch.manual_seed(123)
    model=Candidate3dEncoder(variant,channels=6,candidate_type="JUNCTION")
    x=torch.randn(1,4,8,8,8)
    xyz=torch.stack(torch.meshgrid(
        *(torch.arange(8,dtype=torch.float32) for _ in range(3)),indexing="ij"
    ),dim=0)
    for i,pos in enumerate([[2.,3.,3.],[4.,3.,3.],[3.,5.,3.]]):
        delta=xyz-torch.tensor(pos)[:,None,None,None]
        x[0,i+1]=torch.exp(-(delta**2).sum(dim=0)/(2*1.5**2))
    model.eval()
    with torch.no_grad():
        score,vessel,bridge=model(x,return_structure=True)
        perm=x[:,[0,3,1,2]]
        score2,vessel2,bridge2=model(perm,return_structure=True)
    assert torch.allclose(score,score2,rtol=1e-5,atol=1e-5)
    assert torch.allclose(vessel,vessel2,rtol=1e-5,atol=1e-5)
    assert torch.allclose(bridge,bridge2,rtol=1e-5,atol=1e-5)
    model.train()
    score,vessel,bridge=model(x,return_structure=True)
    loss=(score.sigmoid().mean()+vessel.sigmoid().mean()+bridge.mean())
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_separate_pair_rejects_unintended_third_channel_and_raw_legacy_shape():
    model=Candidate3dEncoder("conv3d_sep_structure",candidate_type="PAIR",channels=6)
    with pytest.raises(ValueError,match="4-channel"):
        model(torch.randn(1,2,8,8,8))
    bad=torch.randn(1,4,8,8,8)
    with pytest.raises(ValueError,match="third endpoint"):
        model(bad)
    good=torch.randn(1,4,8,8,8)
    good[:,3]=0
    result=model(good)
    assert result.shape==(1,) and torch.isfinite(result).all()


def test_separate_heatmap_missing_endpoint_abstains_not_mean_vessel():
    from models.structural_bridge3d import VesselBridgeReadout
    vessel=torch.ones(1,8,8,8)*.99
    tips=torch.zeros(1,3,8,8,8)
    tips[0,0,2,3,3]=1.
    tips[0,1,5,3,3]=1.
    result=VesselBridgeReadout(3,junction_strategy="adaptive_star")(vessel,tips)
    assert result.item()<=1e-5*1.1


def test_gaussian_tip_moments_are_subvoxel_stable_for_close_3d_junction():
    from models.structural_bridge3d import _peak_centers_from_channels
    coords=torch.stack(torch.meshgrid(
        *(torch.arange(12,dtype=torch.float32) for _ in range(3)),indexing="ij"
    ),dim=0)
    reference=torch.tensor([[5.2,6.1,6.0],[7.1,6.1,6.0],[6.0,8.2,6.0]])
    def construct(centers):
        fields=[]
        for p in centers:
            fields.append(torch.exp(-((coords-p[:,None,None,None])**2).sum(0)/(2*1.5**2)))
        return torch.stack(fields)[None]
    original=construct(reference)
    shifted=reference.clone()
    shifted[0,0]+=.2
    moved=construct(shifted)
    p,valid=_peak_centers_from_channels(original,3)
    q,valid2=_peak_centers_from_channels(moved,3)
    assert bool(valid.item() and valid2.item())
    assert torch.allclose(p[0],reference,atol=.04,rtol=0)
    assert torch.allclose(q[0],shifted,atol=.04,rtol=0)
    assert (q-p).abs().max().item()<.3
    assert torch.dist(p[0,0],p[0,1]).item()<3.


@pytest.mark.parametrize("variant",[
    "conv3d_deepsets_sep_structure",
    "anza3d_deepsets_sep_structure",
    "anza3d_noaxial_deepsets_sep_structure",
])
def test_nonlinear_tip_stem_is_symmetric_and_distinguishes_equal_sum_tip_sets(variant):
    """Old 4-channel stem linearly collapsed separate endpoints to one map."""
    torch.manual_seed(11)
    model=Candidate3dEncoder(variant,channels=6,candidate_type="JUNCTION")
    legacy=Candidate3dEncoder(
        "anza3d_sep_structure" if variant.startswith("anza3d") else "conv3d_sep_structure",
        channels=6,candidate_type="JUNCTION",
    )
    q=torch.stack(torch.meshgrid(*(torch.arange(8,dtype=torch.float32)
                                   for _ in range(3)),indexing="ij"))
    field=torch.exp(-((q-3.5)**2).sum(0)/(2*1.3**2))
    a=torch.zeros(1,4,8,8,8)
    b=a.clone()
    a[:,0]=b[:,0]=torch.randn(1,8,8,8)*.25
    for i,m in enumerate((1.,.25,.25)):
        a[:,i+1]=field*m
    for i,m in enumerate((.5,.5,.5)):
        b[:,i+1]=field*m
    assert torch.allclose(a[:,1:].sum(dim=1),b[:,1:].sum(dim=1),atol=1e-7)
    model.eval();legacy.eval()
    with torch.no_grad():
        old_a=legacy.stem(a)
        old_b=legacy.stem(b)
        new_a=model.stem(a)
        new_b=model.stem(b)
        shuffled=model.stem(a[:,[0,3,1,2]])
    assert torch.allclose(old_a,old_b,atol=1e-6,rtol=1e-6)
    assert (new_a-new_b).abs().max().item()>1e-5,(
        "Nonlinear tip moments must retain distinct per-tip multisets"
    )
    assert torch.allclose(new_a,shuffled,atol=1e-6,rtol=1e-6)
    assert model.nonlinear_tips
    assert model.parameter_count()==legacy.parameter_count()+1
    model.train()
    loss=model.stem(a).square().mean()
    loss.backward()
    gain=model.stem.tip_second_moment_gain
    assert gain.grad is not None and torch.isfinite(gain.grad).all()
    assert abs(gain.grad.item())>1e-10
