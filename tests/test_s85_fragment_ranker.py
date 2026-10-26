import json
import numpy as np
import cv2
import pytest

from bubble_chamber.s85_fragment_ranker import (
    FEATURE_NAMES, MODEL_SCHEMA, ARMS, candidate_features,
    fit_train_only, predict_probability, select_extras, model_sha256,
)


def line(y, xmin=20, xmax=155, n=60):
    return np.column_stack([np.linspace(xmin,xmax,n),np.full(n,y)]).astype(float)


def image():
    im=np.full((192,192),206,np.uint8)
    cv2.line(im,(20,66),(155,66),25,2)
    return im


def test_pixel_features_are_finite_and_reverse_invariant():
    img=image()
    a=line(66)
    contacts=[{'score':.92,'original_port':(0,-1),'extra_port':0,'original':0},
              {'score':.72,'original_port':(1,0),'extra_port':-1,'original':1}]
    before=candidate_features(img,a,[line(40),line(90)],contacts,set())
    after=candidate_features(img,a[::-1],[line(40),line(90)],contacts,set())
    assert before.shape==(len(FEATURE_NAMES),)
    assert np.isfinite(before).all()
    np.testing.assert_allclose(before,after,atol=1e-6)
    assert before[0]>candidate_features(img,line(100),[line(40)],[],set())[0]


def test_fail_closed_if_train_labels_too_sparse():
    X=np.ones((12,len(FEATURE_NAMES)))
    y=np.array([1]*3+[0]*9)
    groups=np.arange(12)
    model=fit_train_only(X,y,groups)
    assert not model['trained']
    assert model['reason']=='INSUFFICIENT_UNAMBIGUOUS_TRAIN_LABELS'
    assert predict_probability(model,np.ones(len(FEATURE_NAMES)))==0.
    with pytest.raises(ValueError):fit_train_only(X,np.full(12,2),groups)


def test_train_and_reload_model_deterministic_without_identity_features():
    rng=np.random.default_rng(85)
    X=rng.normal(size=(90,len(FEATURE_NAMES)))
    labels=np.asarray([1 if i%3==0 else 0 for i in range(90)])
    X[:,0]+=labels*.8
    groups=np.repeat(np.arange(30),3)
    first=fit_train_only(X,labels,groups)
    second=fit_train_only(X,labels,groups)
    assert first==second and first['trained']
    assert first['feature_names']==list(FEATURE_NAMES)
    assert 'true_particle' not in json.dumps(first).lower()
    assert 0<predict_probability(first,X[0])<1
    assert predict_probability(json.loads(json.dumps(first)),X[3])==predict_probability(first,X[3])
    assert model_sha256(first)==model_sha256(second)
    with pytest.raises(ValueError):predict_probability({**first,'feature_names':['GT']},X[0])


def test_strict_original_identity_and_no_forced_real_joins():
    original=line(42);extra=line(66)
    model=fit_train_only(np.ones((5,len(FEATURE_NAMES))),[1,0,0,0,0],np.arange(5))
    out=select_extras(image(),[original],[original,extra],{},set(),model,'s85_learned_p85')
    assert out['polylines'][0] is original and out['extra_accepted']==0
    assert not out['physical_auto_joins_authorized']
    with pytest.raises(ValueError):select_extras(image(),[original],[original.copy(),extra],{},set(),model,'baseline')
    with pytest.raises(ValueError):select_extras(image(),[original],[original],{},set(),model,'oracle')
    with pytest.raises(TypeError):select_extras(image(),[original],[original],{},set(),model,'baseline',true_nodes=[])


def test_original_selected_ports_affect_only_graph_features():
    x=line(42);y=line(84);test=line(66)
    contact=[{'score':.93,'original_port':(0,0),'extra_port':0,'original':0}]
    free=candidate_features(image(),test,[x,y],contact,set())
    occupied=candidate_features(image(),test,[x,y],contact,{(0,0)})
    assert free[-2]==1 and occupied[-2]==0
    np.testing.assert_allclose(free[:-2],occupied[:-2])


def test_trained_high_confidence_can_add_only_optional_extra_instances():
    from bubble_chamber.s85_fragment_ranker import FEATURE_SCHEMA
    n=len(FEATURE_NAMES)
    model={'schema':MODEL_SCHEMA,'feature_schema':FEATURE_SCHEMA,
           'feature_names':list(FEATURE_NAMES),'trained':True,
           'scaler_mean':[0.]*n,'scaler_scale':[1.]*n,
           'coef':[0.]*n,'intercept':7.,'physical_auto_joins_authorized':False}
    source=line(40);extra=line(66)
    out=select_extras(image(),[source],[source,extra],{},set(),model,'s85_learned_p95')
    assert out['extra_accepted']==1 and out['extra_rejected']==0
    assert out['polylines'][0] is source and out['polylines'][1] is extra
    assert not out['physical_auto_joins_authorized']
    with pytest.raises(ValueError):predict_probability({**model,'coef':[float('nan')]*n},np.zeros(n))


def test_source_nonlinearity_invalid_image_and_malformed_contacts_rejected():
    img=image();a=line(66);origin=line(40)
    with pytest.raises(ValueError):candidate_features(img.astype(float),a,[origin],[],set())
    with pytest.raises(ValueError):candidate_features(img,a,[origin],[{'score':1.2,'original_port':(0,0),'extra_port':0,'original':0}],set())
    with pytest.raises(ValueError):candidate_features(img,a,[origin],[{'score':.9,'original_port':(2,0),'extra_port':0,'original':2}],set())


def test_image_derived_synthetic_training_smoke_is_only_a_software_check():
    """Contrived drawings; explicitly NOT real H50Z or study performance."""
    from bubble_chamber.s85_fragment_ranker import FEATURE_NAMES
    X=[];y=[];groups=[]
    rng=np.random.default_rng(8500)
    for scene in range(32):
        im=np.full((192,192),int(rng.integers(195,218)),np.uint8)
        true_y=int(rng.integers(48,62))
        cv2.line(im,(20,true_y),(155,true_y),int(rng.integers(12,55)),2)
        for candidate,label in [(line(true_y),1),(line(91),0),(line(118),0)]:
            X.append(candidate_features(im,candidate,[line(29)],[],set()))
            y.append(label)
            groups.append(scene)
    fitted=fit_train_only(np.array(X),np.array(y),np.array(groups))
    assert fitted['trained'] and fitted['train_positives']==32
    assert fitted['train_negatives']==64
    confidence=[predict_probability(fitted,x) for x in X]
    assert np.isfinite(confidence).all()
    assert np.mean([c for c,label in zip(confidence,y) if label==1]) > np.mean(
        [c for c,label in zip(confidence,y) if label==0])
