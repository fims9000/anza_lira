"""Artificial pixel/graph QA only. NOT a H50Z/visualMetro scientific experiment."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from bubble_chamber.s85_fragment_ranker import (
    FEATURE_NAMES,fit_train_only,candidate_features,select_extras,predict_probability
)
from scripts.research.bubble_chamber import s85_train_calibrate_test_learned as s85


def stroke(gray_value):
    a=np.full((192,192),206,np.uint8)
    cv2.line(a,(38,80),(148,80),gray_value,2)
    return a


def candidate():
    return np.column_stack((np.linspace(38,148,45),np.full(45,80))).astype(float)


def test_training_pixels_predicts_selected_extra_and_source_is_immutable(monkeypatch):
    # Synthetic strokes with separate binary labels, never real physical GT.
    original=np.column_stack((np.linspace(30,150,45),np.full(45,140))).astype(float)
    additional=candidate()
    dataset=[];targets=[];groups=[]
    for scene_idx in range(60):
        is_bright = scene_idx%3 != 0
        pix=stroke(206 if is_bright else 10)
        vector=candidate_features(pix,additional,[original],[],set())
        dataset.append(vector)
        targets.append(0 if is_bright else 1)
        groups.append(scene_idx)
    model=fit_train_only(dataset,targets,groups)
    assert model['trained']
    hi=predict_probability(model,dataset[0])
    lo=predict_probability(model,dataset[1])
    assert hi>lo
    # We do not loosen fixed 0.85/0.95 thresholds in S85 to pass a demo.
    simple=select_extras(stroke(10),[original],[original,additional],{},set(),model,'s85_learned_p85')
    assert simple['polylines'][0] is original
    assert simple['extra_accepted'] in (0,1)
    assert not simple['physical_auto_joins_authorized']


def test_cannot_make_fake_scientific_result_without_private_input(tmp_path):
    # No automatic substitution of synthetic local QA for original private data.
    with pytest.raises(FileNotFoundError,match='Private H50Z'):
        s85.run('test',tmp_path/'none',tmp_path/'none',
                train_models=tmp_path/'train.json',calibration=tmp_path/'cal.json')
