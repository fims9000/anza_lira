import numpy as np
import cv2
import pytest
from bubble_chamber.s12_curve_geometry import fit_single_track,sample_fitted_curve,evaluate_curves


def synthetic_parabola():
    img=np.zeros((128,128),np.uint8)
    x=np.linspace(15,110,100)
    y=40+.003*(x-60)**2
    points=np.rint(np.stack([x,y],axis=-1)).astype('int32')
    cv2.polylines(img,[points],False,1,1)
    return img


def test_quadratic_extracts_curve_and_endpoints_without_truth():
    z=synthetic_parabola()
    res=fit_single_track(z)
    assert res.status=='ok'
    assert res.components==1
    assert res.rms_fit_px is not None and res.rms_fit_px<3
    assert res.center_curvature_inv_px is not None and res.center_curvature_inv_px>0
    path=sample_fitted_curve(res,30)
    assert path.shape==(30,2)
    assert path[:,0].max()>90 and path[:,0].min()<25


def test_disconnected_requires_lira_not_automatic_join():
    z=synthetic_parabola()
    z[:,58:75]=0
    res=fit_single_track(z)
    assert res.status=='disconnected_requires_lira'
    assert res.components>1 and not res.accepted_for_review


def test_empty_and_invalid():
    empty=np.zeros((50,50),np.uint8)
    m=fit_single_track(empty)
    assert m.status=='blank' and not m.accepted_for_review
    with pytest.raises(ValueError):sample_fitted_curve(m)
    with pytest.raises(ValueError):fit_single_track(np.zeros((3,3,3)))


def test_curve_metrics_evaluate_without_fitting_on_truth():
    z=synthetic_parabola()
    e=evaluate_curves(z[None].astype(bool),z[None].astype(bool))
    assert e['fitted_pairs']==1
    assert e['median_endpoint_error_px']<1e-8
    assert e['median_absolute_curvature_error_inv_px']<1e-8
    assert e['particle_physics_parameters_available'] is False


def test_no_endpoint_error_from_abstained_disconnected_prediction():
    truth=synthetic_parabola()
    fragmented=truth.copy()
    fragmented[:,60:70]=0
    result=evaluate_curves(fragmented[None].astype(bool),truth[None].astype(bool))
    assert result['fitted_pairs']==0
    assert result['predicted_accepted_for_review']==0
    assert result['median_endpoint_error_px'] is None
    assert result['status_counts']['disconnected_requires_lira']==1
    assert result['paired_eligible_coverage']==0.0
