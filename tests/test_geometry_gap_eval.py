import numpy as np
from bubble_chamber.geometry_gap_eval import _predict, evaluate_geometry
from bubble_chamber.visualmetro_import import SCHEMA


def test_linear_points_recovered_exactly():
    p=np.stack((np.arange(9)*3.,np.arange(9)*7.),1)
    pred=_predict(p,3,2)
    assert np.allclose(pred['linear'],p[3:5])
    assert np.allclose(pred['local_quadratic'],p[3:5])
    assert np.allclose(pred['local_hermite'],p[3:5])


def test_quadratic_reconstructs_curved_points():
    x=np.arange(12.,dtype=float)
    y=15+3*x+0.25*x*x
    p=np.stack([x,y],1)
    pred=_predict(p,5,2)
    assert np.allclose(pred['local_quadratic'],p[5:7],atol=1e-9)
    assert np.linalg.norm(pred['linear']-p[5:7])>0.5


def test_eval_does_not_fabricate_photo_ground_truth():
    x=[[i,2*i+0.03*i*i] for i in range(10)]
    m={"schema":SCHEMA,"frames":[{"film_id":"070","frame_id":"070_008","tracks":[
        {"track_id":"e1t01","n_points":10,"measured_pixel_xy":{str(v):x for v in (1,2,3)}}]}]}
    result=evaluate_geometry(m)
    assert result['evaluation_traces']==3
    assert result['no_real_segmentation_metric'] is True
    assert result['per_method']['local_quadratic']['mean_px'] < result['per_method']['linear']['mean_px']
