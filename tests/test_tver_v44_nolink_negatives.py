"""Pure V44 TRAIN-only candidate negative substitution tests."""
import numpy as np
import pytest
from scripts.tver_v44_nolink_negatives import swap_nolink_negatives


class DummyBaseline:
    def predict_proba(self,x):
        p=np.clip(np.asarray(x)[:,0]/50+.5,.01,.99)
        return np.column_stack([1-p,p])


def fixture():
    f=[];y=[];scenes=[]
    for sid in range(20):
        good=[sid+10.]+[0.]*11
        bad=[sid+5.]+[0.]*11
        null=[sid+8.]+[0.]*11
        scenes.extend([
          {"id":str(sid),"section":sid,"kind":"LINK","proposals":[
              {"y":1,"features":good},{"y":0,"features":bad}]},
          {"id":str(sid),"section":sid,"kind":"NO_LINK","proposals":[
              {"y":0,"features":null}]}
        ])
        f.extend([good,bad]);y.extend([1,0])
    return scenes,np.asarray(f),np.asarray(y)


def test_same_label_budget_half_and_all():
    scenes,f,y=fixture()
    half,h=swap_nolink_negatives(scenes,f,y,DummyBaseline(),"half")
    full,g=swap_nolink_negatives(scenes,f,y,DummyBaseline(),"all")
    assert h["replaced_no_link_negatives"]==10
    assert g["replaced_no_link_negatives"]==20
    assert len(half)==len(full)==len(f)
    assert np.array_equal(half[::2],f[::2])
    assert np.array_equal(full[::2],f[::2])
    assert np.array_equal(half[3::4],f[3::4])


def test_fails_closed_when_source_or_label_drifts():
    scenes,f,y=fixture()
    with pytest.raises(ValueError,match="labels"):
        swap_nolink_negatives(scenes,f,np.zeros_like(y),DummyBaseline(),"all")
    scenes[1]["proposals"][0]["y"]=1
    with pytest.raises(ValueError,match="NO_LINK"):
        swap_nolink_negatives(scenes,f,y,DummyBaseline(),"all")
    with pytest.raises(ValueError):
        swap_nolink_negatives(scenes,f,y,DummyBaseline(),"unknown")
