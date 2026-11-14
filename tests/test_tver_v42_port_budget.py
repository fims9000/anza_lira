"""Pure fixed-budget/honest candidate matching tests."""
import pytest
from scripts.tver_v42_port_budget import evaluate_scenes,summarize,ALLOCATIONS


def proposal(prob,gray,rect,t,*,max_sources,per_source_k):
    bank=[{"source":(float(i*10),0.),"target":(float(i*10+4),0.),
           "geometry":float(-i),"features":[0.]*12} for i in range(max_sources*per_source_k)]
    return {"proposals":bank[:48],"n_proposals_before_cap":len(bank),
            "n_eligible_source_ports_before_cap":15,
            "source_count":max_sources,"hard_branches":8,"status":"OK"}


def sample_scene():
    return {"visible_prob":None,"gray":None,"rectangle":(0,0,0,0),
            "proposals":proposal(None,None,None,.65,max_sources=4,per_source_k=12)["proposals"],
            "kind":"LINK","true_ends":((100,0),(104,0)),"origin":(0,0),
            "id":"test-gap","section":40}


def test_equal_max_budget_exact_original_and_label_free():
    scene=sample_scene()
    rows=evaluate_scenes("REL_CAL_ALREADY_VIEWED",[scene],proposal,
                         lambda p,ends,origin:p["source"]==ends[0] and p["target"]==ends[1])
    assert len(rows)==len(ALLOCATIONS)
    assert all(r["n_candidates"]<=48 for r in rows)
    assert all(r["n_eligible_source_ports_before_cap"]==15 for r in rows)
    assert all(r["has_true_candidate"] for r in rows)


def test_fail_if_baseline_not_identical_or_oracle_in_pair():
    scene=sample_scene();scene["proposals"]=[{"source":(9.,0.),"target":(10.,0.)}]
    with pytest.raises(RuntimeError,match="Frozen"):
        evaluate_scenes("TRAIN",[scene],proposal,lambda p,ends,origin:False)
    def leaky(p,g,r,t,*,max_sources,per_source_k):
        x=proposal(p,g,r,t,max_sources=max_sources,per_source_k=per_source_k)
        x["proposals"][0]["y"]=1
        return x
    with pytest.raises(ValueError,match="Oracle"):
        evaluate_scenes("TRAIN",[sample_scene()],leaky,lambda p,ends,origin:False)
