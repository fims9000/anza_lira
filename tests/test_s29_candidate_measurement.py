import cv2,numpy as np,pytest
from bubble_chamber.s29_candidate_measurement import parametric_quadratic_fit,photo_line_profile,measure_candidate


def test_vertical_quadratic_fit_stable_not_y_of_x():
    u=np.linspace(0,1,100)
    x=75+1.6*u+10*u*u
    y=20+158*u-2*u*u
    fit=parametric_quadratic_fit(np.stack([x,y],axis=1))
    assert fit['rms_centerline_residual_px']<.1
    assert fit['pixel_arclength']>150
    assert abs(fit['source_start_xy'][0]-75)<.001
    assert fit['coordinate_frame']=='2D_CROP_PIXEL' if 'coordinate_frame' in fit else True


def test_line_width_on_synthetic_dark_ridge():
    gray=np.full((256,256),195,dtype=np.uint8)
    pts=np.array([[x,128] for x in range(15,238)],dtype=int)
    cv2.polylines(gray,[pts.reshape(-1,1,2)],False,40,3,cv2.LINE_8)
    result=photo_line_profile(gray,pts.tolist())
    assert result['fraction_with_dark_center']>.9
    assert result['median_apparent_width_px']>=2
    assert result['physical_ionization_deduction_allowed'] is False


def test_no_ridge_means_no_physical_width_inferred():
    img=np.full((256,256),190,np.uint8)
    points=[[x,100] for x in range(25,225)]
    d=photo_line_profile(img,points)
    assert d['profiles_with_dark_center']==0
    assert d['median_apparent_width_px'] is None
    assert measure_candidate(img,{'path_points_original_xy':points})['expert_approved'] is False


def test_invalid_source_or_bad_coords_rejected():
    with pytest.raises(ValueError):parametric_quadratic_fit([[3,1],[4,2]])
    with pytest.raises(ValueError):photo_line_profile(np.zeros((64,64,3),np.uint8),[[i,12] for i in range(15)])
    with pytest.raises(ValueError):parametric_quadratic_fit([[float('nan'),i] for i in range(16)])
