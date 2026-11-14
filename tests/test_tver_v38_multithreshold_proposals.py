"""Pure tests: label-free proposals, invariant budget, duplicates, analysis."""
import numpy as np
import pytest

from scripts.tver_v38_multithreshold_proposals import (
    CAP,METHODS,proposal_banks,merge_fixed_budget,_rows_for_scenes,
    summarize,same_pair)


def item(a,b,geometry=1.):
    return {"source":(a,0.),"target":(b,0.),"geometry":geometry,
            "features":[0.]*12}


def mocked_propose(prob, gray, rect, threshold):
    bank={.45:[item(1,5,.8),item(10,15,.6)],
          .55:[item(1,5,.8),item(20,25,.7)],
          .65:[item(1,5,.8),item(40,45,.5)],
          .75:[item(60,65,.9)]}[threshold]
    return {"proposals":bank,"status":"OK","hard_branches":2,"source_count":1}


def test_label_free_first_and_fair_cap():
    banks=proposal_banks(np.zeros((2,2)),np.zeros((2,2)),(0,0,0,0),mocked_propose)
    for name,thresholds in METHODS.items():
        p=merge_fixed_budget(banks,thresholds,cap=CAP)
        assert len(p)<=CAP
        assert len(p)==len({(tuple(z['source']),tuple(z['target'])) for z in p})
        assert all('y' not in z for z in p)
    assert len(merge_fixed_budget(banks,(.65,)))==2
    assert len(merge_fixed_budget(banks,(.65,.55)))==3
    assert same_pair(item(1,5),item(5,1))


def test_near_duplicate_and_fixed_48_pairs():
    raw=lambda ts: {t:{'proposals':[item(k*10+t*0.01,k*10+5+t*0.01,1.)
             for k in range(70)],'hard_branches':70,'source_count':1,'status':'OK'}
                    for t in ts}
    banks=raw((.65,.55))
    assert len(merge_fixed_budget(banks,(.65,.55)))==48
    assert len(merge_fixed_budget(banks,(.65,)))==48


def test_reject_oracle_encoded_candidates():
    def bad(p,g,r,t):
        z=mocked_propose(p,g,r,t)
        z['proposals'][0]={**z['proposals'][0], 'y':1}
        return z
    with pytest.raises(ValueError,match='ground-truth'):
        proposal_banks(None,None,None,bad)


def test_baseline_exact_identity_and_posthoc_rescue_accounting():
    scenes=[{"visible_prob":np.zeros((2,2)),"gray":np.zeros((2,2)),
             "rectangle":(0,0,0,0),"proposals":mocked_propose(0,0,0,.65)['proposals'],
             "kind":"LINK","true_ends":((20,0),(25,0)),"origin":(0,0),
             "id":"gap1","section":41}]
    valid=lambda p,ends,origin: tuple(p['source'])==tuple(ends[0]) and tuple(p['target'])==tuple(ends[1])
    rows=_rows_for_scenes('REL_CAL_ALREADY_VIEWED',scenes,mocked_propose,valid)
    summ=summarize(rows)
    b=summ['REL_CAL_ALREADY_VIEWED/original_065']
    new=summ['REL_CAL_ALREADY_VIEWED/union_055_065']
    assert b['correct_candidate_available']==0
    assert new['correct_candidate_available']==1
    assert new['rescue_vs_original']==1
    assert new['candidate_budget']==48
    assert b['geometry_top1_correct']==0