"""V25 regression contracts on fabricated, matched hard synthetic worlds.

Not a medical / natural seismic benchmark. No V25 locked TEST outcomes in
these tests; verifies exact source modalities and CNN inputs before running.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from scripts.tver_v24_switched_neighbor_synthetic import render_hard_v24
from scripts.tver_v25_equal_hard_training import BASE,N,_build,METHODS
from scripts.tver_v25_cnn import (
    Tiny2p5DCNN,encode_observed_worlds,train_image_statistics,
)


def test_v25_train_cal_and_unseen_test_have_disjoint_base_groups():
    sets={s:set(range(BASE[s],BASE[s]+N[s])) for s in BASE}
    assert N=={"train":200,"cal":100,"test":160}
    for s in sets:
        for t in sets:
            if s!=t:
                assert sets[s].isdisjoint(sets[t])
    assert len(METHODS)==8


def test_v25_equal_hard_negative_in_each_split():
    for s in BASE:
        r=render_hard_v24(0,base_seed=BASE[s],total_groups=N[s])
        assert len(r)==3
        assert {x["label"] for x in r}=={0,1,2}
        assert all(np.array_equal(x["central"],r[0]["central"]) for x in r)
        assert all(x["candidates"]==r[0]["candidates"] for x in r)
        n=next(x for x in r if x["label"]==2)
        positive=[x for x in r if x["label"]!=2]
        minus=[x for x in positive if np.array_equal(
            x["adjacent"][0],n["adjacent"][0])]
        plus=[x for x in positive if np.array_equal(
            x["adjacent"][1],n["adjacent"][1])]
        assert len(minus)==len(plus)==1
        assert minus[0]["label"]!=plus[0]["label"]


def test_raw_2d_cnn_is_observationally_blind_to_all_three_worlds():
    r=render_hard_v24(3,base_seed=BASE["train"],total_groups=200)
    mu,sd=train_image_statistics(r)
    two=encode_observed_worlds(r,mode="central_2d",mean=mu,std=sd)
    three=encode_observed_worlds(r,mode="neighbors_25d",mean=mu,std=sd)
    assert two.shape==three.shape==(3,6,64,64)
    assert np.array_equal(two[0],two[1])
    assert np.array_equal(two[0],two[2])
    assert np.any(three[0]!=three[2])
    assert np.any(three[1]!=three[2])
    assert np.array_equal(three[0,3:],three[2,3:])


def test_CNN_input_never_reads_hidden_label_base_or_world():
    r=render_hard_v24(3,base_seed=BASE["train"],total_groups=200)
    mu,sd=train_image_statistics(r)
    before=encode_observed_worlds(r,mode="neighbors_25d",mean=mu,std=sd)
    altered=copy.deepcopy(r)
    for c in altered:
        c["label"]=17
        c["world"]=999
        c["base"]=-1
    after=encode_observed_worlds(altered,mode="neighbors_25d",mean=mu,std=sd)
    np.testing.assert_array_equal(before,after)


def test_three_class_CNN_output_and_forward_are_finite():
    torch.manual_seed(12)
    net=Tiny2p5DCNN()
    x=torch.randn(3,6,64,64)
    with torch.no_grad():
        y=net(x)
    assert y.shape==(3,3)
    assert bool(torch.isfinite(y).all())


def test_cnn_uses_frozen_optimizer_budget():
    from scripts.tver_v25_cnn import (
        N_EPOCHS,BATCH,RATE,WEIGHT_DECAY,TORCH_SEEDS)
    assert N_EPOCHS==12
    assert BATCH==48
    assert RATE==.001 and WEIGHT_DECAY==.01
    assert TORCH_SEEDS==(11,23)
