"""V24 synthetic cross-slice switching NO_LINK contracts, no outcomes opened."""
from __future__ import annotations

import copy
import hashlib
import numpy as np

from scripts.tver_v24_switched_neighbor_synthetic import (
    BASE,N,render_hard_v24,_visible_scores,
)
from scripts.tver_v23_25d_counterfactual_synthetic import (
    SEED_BASE,N_BASE,
)


def test_hard_negative_identical_central_pixels_and_opposite_true_single_near_marginals():
    for ix in (0,1,2,7,23):
        rows=render_hard_v24(ix)
        assert len(rows)==3
        assert {r["label"] for r in rows}=={0,1,2}
        assert all(np.array_equal(r["central"],rows[0]["central"]) for r in rows)
        assert all(r["source"]==rows[0]["source"] for r in rows)
        assert all(r["candidates"]==rows[0]["candidates"] for r in rows)
        negative=rows[2]
        same_minus=[p["label"] for p in rows[:2]
                    if np.array_equal(p["adjacent"][0],
                                      negative["adjacent"][0])]
        same_plus=[p["label"] for p in rows[:2]
                   if np.array_equal(p["adjacent"][1],
                                     negative["adjacent"][1])]
        assert len(same_minus)==len(same_plus)==1
        assert same_minus[0]!=same_plus[0]
        assert np.array_equal(negative["far"],rows[0]["far"])


def test_single_slice_exact_negative_matching_stops_marginal_NO_LINK_shortcuts():
    rows=render_hard_v24(3)
    negative=rows[2]
    for z in (0,1):
        counterpart=next(r for r in rows[:2]
                         if np.array_equal(
                             r["adjacent"][z],negative["adjacent"][z]))
        # This negative is observationally identical in the selected slice.
        a=hashlib.sha256(counterpart["adjacent"][z].tobytes()).hexdigest()
        b=hashlib.sha256(negative["adjacent"][z].tobytes()).hexdigest()
        assert a==b
    # Yet BOTH slices together are different from either positive world.
    assert not any(np.array_equal(negative["adjacent"],
                                  r["adjacent"]) for r in rows[:2])


def test_new_consensus_observes_only_images_not_hidden_world_label():
    rows=render_hard_v24(6)
    for a in rows:
        f=_visible_scores(a)
        modified=copy.deepcopy(a)
        modified["label"]=12
        modified["world"]=999
        modified["base"]=0
        f2=_visible_scores(modified)
        for k in f:
            np.testing.assert_array_equal(f[k],f2[k])
        assert f["consensus_v24"].shape==(2,)
        assert np.isfinite(f["consensus_v24"]).all()
    # Central-only features cannot distinguish all 3 equal inputs.
    x=[_visible_scores(r)["center_features"] for r in rows]
    assert np.array_equal(x[0],x[1])
    assert np.array_equal(x[0],x[2])


def test_unseen_v24_base_ids_do_not_intersect_any_v23_partition():
    new=set(range(BASE,BASE+N))
    for split,seed in SEED_BASE.items():
        previous=set(range(seed,seed+N_BASE[split]))
        assert new.isdisjoint(previous)
