"""V50 source-conditioned neural path decoder: no-label inference and DP tests.

All fixtures synthetic; real geological calibration is evaluated separately in
the frozen GitHub Actions source workflow.
"""
import numpy as np
import pytest
import torch

from scripts import tver_v50_conditional_path as m


def data_scene(n=50,positive=True):
    p=np.ones((96,96),np.float32)*.03
    p[:,46:48]=.94
    p[43:55,42:52]=0
    gray=np.linspace(.25,.75,96*96,dtype=np.float32).reshape(96,96)
    gray[:,46:48]+=.10
    prop={"source":(30.,47.),"target":(65.,47.),"geometry":-.5,"y":int(positive)}
    other={"source":(30.,42.),"target":(65.,43.),"geometry":-.8,"y":0}
    return {
        "gray":gray,"visible_prob":p,"rectangle":(43,55,42,52),
        "origin":(0,0),"section":n,"id":str(n),"kind":"LINK",
        "proposals":[prop,other] if positive else [other],
        "hidden_centerline":[(y,47) for y in range(43,55)],
        "true_ends":((42,47),(56,47))
    }


def test_observed_path_input_never_uses_hidden_truth():
    s=data_scene()
    x,pos=m.observed_features(s,s["proposals"][0])
    assert x.shape==(7,48,48) and np.isfinite(x).all()
    assert x[2].sum()==12*10
    gt=m.target_inside_rectangle(s,pos)
    assert gt.shape==(48,48) and gt.sum()>0
    no_truth={k:v for k,v in s.items()
              if k not in ("hidden_centerline","true_ends","origin","section","id")}
    x2,_=m.observed_features(no_truth,s["proposals"][0])
    assert np.array_equal(x,x2)
    s["hidden_centerline"]=[(y,43) for y in range(43,55)]
    x3,_=m.observed_features(s,s["proposals"][0])
    assert np.array_equal(x,x3)


def test_ribbon_can_follow_a_curved_high_evidence_path():
    heat=np.ones((96,96),np.float32)*.01
    for y in range(20,76):
        x=45+round(4*np.sin(np.pi*(y-20)/55))
        heat[y,x-1:x+2]=.95
    route=m.ribbon_dp_route(heat,(20,45),(75,45),radius=8)
    assert len(route)>35
    assert route[0]==(20,45) and route[-1]==(75,45)
    assert any(47<=x<=50 and 38<=y<=58 for y,x in route)


def test_train_save_and_predict_with_no_test_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"EPOCHS",2)
    scenes=[data_scene(i) for i in range(40)]
    model,receipt=m.fit_path_head(scenes,tmp_path)
    assert receipt["positive_path_examples"]==40
    assert receipt["same_scene_hard_negative_examples"]==40
    assert receipt["epochs"]==2
    assert (tmp_path/receipt["checkpoint_filename"]).exists()
    assert receipt["model_parameters"]>1000
    s=data_scene(2)
    scores=m.path_heatmap(model,s,s["proposals"][0])
    assert scores.shape==(96,96) and np.isfinite(scores).all()
    route,meta=m.decode_candidate(model,s,s["proposals"][0],mode="learned_ribbon")
    assert len(route)>10 and meta["no_hidden_endpoints_used"]
    s.pop("hidden_centerline");s.pop("true_ends")
    s["proposals"][0].pop("y")
    second,_=m.decode_candidate(model,s,s["proposals"][0],mode="learned_ribbon")
    assert route==second


def test_synthetic_data_and_bad_endpoints_fail_closed():
    s=data_scene()
    bad={**s,"visible_prob":np.full((96,96),np.nan,np.float32)}
    with pytest.raises(ValueError):
        m.observed_features(bad,s["proposals"][0])
    with pytest.raises(ValueError):
        m.observed_features(s,{"source":(1.,1.),"target":(1.,1.)})
    s["hidden_centerline"]=[]
    with pytest.raises(ValueError):
        m.target_inside_rectangle(s,(24,24))
    with pytest.raises(RuntimeError):
        m.build_train_dataset([data_scene(2)])
