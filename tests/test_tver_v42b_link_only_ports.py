"""Pure V42b source-port allocation test without real Thebe arrays."""
import pytest
from scripts.tver_v42b_link_only_ports import compare_positive_scenes,ALLOCATIONS,MAX_PAIRS


def fake_scene():
    return {
      "section":40,"id":"40:0","kind":"LINK",
      "proposals":[{"source":(10.,10.),"target":(20.,20.),
                    "features":[0.]*12,"geometry":-.1,"y":1}],
      "status":"OK","source_count":4,"branch_count":10,
      "visible_prob":None,"gray":None,"rectangle":(0,0,0,0),
      "origin":(0,0),"true_ends":((10.,10.),(20.,20.)),
    }


def fake_propose(prob,gray,rect,t,*,max_sources,per_source_k):
    assert max_sources*per_source_k==48
    return {"proposals":[{"source":(10.,10.),"target":(20.,20.),
                          "geometry":-.1,"features":[0.]*12}],
            "source_count":max_sources,
            "n_eligible_source_ports_before_cap":12,
            "hard_branches":10,"status":"OK"}


def test_fair_original_and_more_sources_on_same_correct_pair():
    r=compare_positive_scenes([fake_scene()],fake_propose,
                             lambda p,e,o: p["source"]==e[0] and p["target"]==e[1])
    assert len(r)==len(ALLOCATIONS)
    assert all(q["has_true_candidate"] and q["geometric_top1_correct"] for q in r)
    assert all(q["n_candidates"]<=MAX_PAIRS for q in r)
    assert all(q["n_source_ports_eligible_before_cap"]==12 for q in r[1:])


def test_no_oracle_target_in_proposed_feature():
    def leaky(*args,**kwargs):
        r=fake_propose(*args,**kwargs)
        r["proposals"][0]["y"]=1
        return r
    with pytest.raises(ValueError,match="Expert truth"):
        compare_positive_scenes([fake_scene()],leaky,lambda p,e,o:True)
    fake=fake_scene();fake["kind"]="NO_LINK"
    with pytest.raises(ValueError):
        compare_positive_scenes([fake],fake_propose,lambda p,e,o:False)
