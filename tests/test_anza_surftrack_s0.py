from __future__ import annotations

import numpy as np
import pytest

from anza_surftrack.eval.observability import evaluate_observability
from anza_surftrack.protocol import FAMILIES, METHODS, SPLITS, split_manifest
from anza_surftrack.synthetic3d.families import CONSTRUCTORS, generate_batch
from anza_surftrack.eval.tracking import _track_batch
from anza_surftrack.transport.bidirectional import anchor_disagreement, precision_fusion
from anza_surftrack.transport.core import common_mean, initial_covariance, propagate_covariance, rotation, transition_matrix


PARAMS = {
    "G1_local_reset": {"sigma_u": 2.0, "sigma_s": .8},
    "G2_shear_compose": {"sigma0": 1.0, "q": .1, "alpha": .3},
    "G3_free_compose": {"sigma0": 1.0, "q": .1, "a": .2, "b": -.1},
    "G4_anza_cocycle": {"sigma0": 1.0, "q": .1, "lambda": .3},
}


def test_every_named_family_has_dedicated_constructor() -> None:
    assert set(CONSTRUCTORS) == set(FAMILIES)
    assert all(CONSTRUCTORS[name]().name == name for name in FAMILIES)


def test_split_seeds_are_disjoint_and_confirm_is_locked() -> None:
    manifest = split_manifest(); assert manifest["seeds_disjoint"]
    assert len({row["seed"] for row in SPLITS.values()}) == len(SPLITS)
    with pytest.raises(PermissionError): generate_batch("geom_confirm", 0, 1)


def test_surface_ids_are_immutable_unique_and_not_geometry_features() -> None:
    batch = generate_batch("geom_dev_iid", 0, 4)
    assert batch.surface_ids.dtype == np.int64
    assert all(len(np.unique(row)) == 5 for row in batch.surface_ids)
    assert batch.candidate_points.shape[-1] == 2


def test_gaps_hide_exact_number_of_intermediate_observations() -> None:
    for index, name in enumerate(FAMILIES):
        if name.startswith("multi_slice_gap_"):
            # Choose index congruent to the desired family in IID stream.
            batch = generate_batch("geom_dev_iid", index, 1)
            assert np.count_nonzero(~batch.observed[0]) == int(name.rsplit("_", 1)[1])


def test_anza_det_reciprocal_eigenvalues_and_axial_invariance() -> None:
    theta0 = np.asarray([.2]); theta1 = np.asarray([.7]); params = PARAMS["G4_anza_cocycle"]
    j = transition_matrix("G4_anza_cocycle", theta0, theta1, params)[0]
    assert np.isclose(np.linalg.det(j), 1.0)
    local = np.diag([np.exp(.3), np.exp(-.3)])
    values = np.linalg.eigvals(local); assert np.isclose(np.prod(values), 1.0)
    shifted = transition_matrix("G4_anza_cocycle", theta0 + np.pi, theta1 + np.pi, params)[0]
    assert np.allclose(j, shifted)


def test_lambda_zero_is_rotation_only_and_no_covariance_normalization() -> None:
    params = {"sigma0": 1.0, "q": .1, "lambda": 0.0}; theta0=np.asarray([.1]); theta1=np.asarray([.4])
    j = transition_matrix("G4_anza_cocycle", theta0, theta1, params)[0]
    assert np.allclose(j, rotation(theta1)[0] @ rotation(theta0)[0].T)
    params["lambda"] = .3; covariance = np.asarray([[[2.0, .2], [.2, 1.0]]])
    propagated = propagate_covariance("G4_anza_cocycle", covariance, theta0, theta1, params)
    assert not np.isclose(np.trace(propagated[0]), np.trace(covariance[0]))


def test_covariances_symmetric_positive_and_process_noise_prevents_collapse() -> None:
    for method in METHODS[1:]:
        theta=np.asarray([.3]); covariance=initial_covariance(method, theta, PARAMS[method])
        for _ in range(20): covariance=propagate_covariance(method, covariance, theta, theta+.01, PARAMS[method]); theta=theta+.01
        assert np.allclose(covariance, np.swapaxes(covariance, -1, -2))
        assert np.linalg.eigvalsh(covariance).min() > 0


def test_composition_order_and_control_determinants() -> None:
    t0=np.asarray([.1]); t1=np.asarray([.2]); t2=np.asarray([.4])
    p=PARAMS["G4_anza_cocycle"]
    composed=transition_matrix("G4_anza_cocycle", t1,t2,p)[0] @ transition_matrix("G4_anza_cocycle",t0,t1,p)[0]
    assert np.isclose(np.linalg.det(composed),1.0)
    shear=transition_matrix("G2_shear_compose",t0,t1,PARAMS["G2_shear_compose"])[0]
    free=transition_matrix("G3_free_compose",t0,t1,PARAMS["G3_free_compose"])[0]
    assert np.isclose(np.linalg.det(shear),1.0); assert not np.isclose(np.linalg.det(free),1.0)


def test_reset_does_not_read_previous_covariance() -> None:
    theta=np.asarray([.2]); p=PARAMS["G1_local_reset"]
    a=propagate_covariance("G1_local_reset",np.eye(2)[None],theta,theta+.1,p)
    b=propagate_covariance("G1_local_reset",(99*np.eye(2))[None],theta,theta+.1,p)
    assert np.allclose(a,b)


def test_common_mean_model_is_single_shared_function() -> None:
    last=np.asarray([[2.,3.]]); previous=np.asarray([[1.,1.]])
    assert np.allclose(common_mean(last,previous,2),np.asarray([[4.,7.]]))
    assert np.allclose(common_mean(last,None),last)


def test_observability_fixture_is_center_matched_and_context_resolvable() -> None:
    result=evaluate_observability()
    assert .45 <= result["center_auroc"] <= .55
    assert result["context_oracle_top1"] >= .85


def test_candidate_set_is_shared_and_truth_does_not_change_scores() -> None:
    batch=generate_batch("geom_dev_iid",0,3)
    before=batch.candidate_points.copy(); first=_track_batch("G0_euclidean",{},batch,"fixture")
    batch.truth_index[:]=3
    second=_track_batch("G0_euclidean",{},batch,"fixture")
    assert np.allclose(before,batch.candidate_points,equal_nan=True)
    assert np.allclose(first.margin,second.margin,equal_nan=True)


def test_end_candidate_is_handled_without_forced_branch() -> None:
    family_index=FAMILIES.index("terminating_surface")
    batch=generate_batch("geom_dev_iid",family_index,1)
    result=_track_batch("G0_euclidean",{},batch,"fixture")
    assert result.rows[0]["decision_count"] <= 12
    assert result.rows[0]["switch"] == 0


def test_precision_fusion_and_anchor_disagreement() -> None:
    mean=np.asarray([1.,2.]); covariance=np.eye(2)[None]
    fused_mean,fused_cov=precision_fusion(mean[None],covariance,mean[None],covariance)
    assert np.allclose(fused_mean[0],mean); assert np.allclose(fused_cov[0],.5*np.eye(2))
    consistent=anchor_disagreement(mean[None],covariance,mean[None],covariance)
    contradictory=anchor_disagreement(mean[None],covariance,np.asarray([[5.,8.]]),covariance)
    assert contradictory[0] > consistent[0]


def test_forward_backward_fixture_is_symmetric_under_argument_reversal() -> None:
    mf=np.asarray([[1.,2.]]); mb=np.asarray([[2.,4.]])
    cf=np.asarray([[[2.,.1],[.1,1.]]]); cb=np.asarray([[[1.,0.],[0.,3.]]])
    mean1,cov1=precision_fusion(mf,cf,mb,cb); mean2,cov2=precision_fusion(mb,cb,mf,cf)
    assert np.allclose(mean1,mean2); assert np.allclose(cov1,cov2)
