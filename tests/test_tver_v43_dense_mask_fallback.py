"""Pure V43 fallback unit tests: no real geological data or model."""
import numpy as np
import pytest
from scripts.tver_v43_dense_mask_fallback import adaptive_threshold,candidate_banks,case_rows,METHODS,BASE


def fake_propose(prob,gray,rect,threshold):
    if float((prob>=threshold).mean())>.38:
        return {"proposals":[],"source_count":0,"hard_branches":0,
                "status":"DEGENERATE_DENSE_PREDICTED_MASK"}
    if int((prob>=threshold).sum())<12:
        return {"proposals":[],"source_count":0,"hard_branches":0,
                "status":"NO_PREDICTED_FAULT_PIXELS"}
    return {"proposals":[{"source":(2.,5.),"target":(6.,9.),
                         "geometry":-.4,"features":[0.]*12}],
            "source_count":1,"hard_branches":2,"status":"OK"}


def scene(prob):
    baseline=fake_propose(prob,None,None,BASE)
    return {"visible_prob":prob,"gray":np.zeros((96,96)),
            "rectangle":(0,0,0,0),"proposals":baseline["proposals"],
            "status":baseline["status"],"kind":"LINK",
            "true_ends":((2.,5.),(6.,9.)),"origin":(0,0),
            "id":"s40gap0","section":40}


def test_rescue_dense_and_identity_on_non_dense():
    p=np.linspace(.2,1.,96*96,dtype=np.float32).reshape(96,96)
    s=scene(p)
    assert s["status"]=="DEGENERATE_DENSE_PREDICTED_MASK"
    banks=candidate_banks(s,fake_propose)
    assert not banks["baseline_065"][0]["proposals"]
    assert all(banks[m][0]["proposals"] for m in METHODS if m!="baseline_065")
    rows=case_rows("REL_CAL_ALREADY_VIEWED",[s],fake_propose,
                   lambda p,ends,origin:p["source"]==ends[0] and p["target"]==ends[1])
    assert sum(r["has_true_candidate"] for r in rows)==3
    assert all(r["n_candidates"]<=48 for r in rows)
    q=np.full((96,96),.01,np.float32)
    q[:10,:30]=.9
    unchanged=candidate_banks(scene(q),fake_propose)
    assert all(unchanged[m][0] is unchanged["baseline_065"][0] for m in METHODS)


def test_quantile_ties_nan_and_baseline_fail_closed():
    p=np.ones((96,96),np.float32)
    threshold=adaptive_threshold(p,.7)
    assert threshold>1 and not (p>=threshold).any()
    with pytest.raises(ValueError):
        adaptive_threshold(np.full((96,96),np.nan),.8)
    with pytest.raises(ValueError):
        adaptive_threshold(np.zeros((96,96)),1.)
    wrong=scene(np.linspace(.2,1.,96*96,dtype=np.float32).reshape(96,96))
    wrong["proposals"]=[{"source":(0.,0.),"target":(3.,2.)}]
    with pytest.raises(RuntimeError,match="not reproduced"):
        candidate_banks(wrong,fake_propose)
