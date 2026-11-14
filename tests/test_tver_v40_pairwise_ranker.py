"""V40 paired-rank loss tests; entirely synthetic fixtures."""
import numpy as np
import pytest
from scripts.tver_v40_pairwise_ranker import (
    fit_same_budget_pair_ranker,pair_rank_scores,evaluate_paired)

def test_training_contrasts_change_pair_order():
    rng=np.random.default_rng(17)
    f=rng.normal(size=(40,12))
    f[::2,0]+=3
    labels=np.tile([1,0],20)
    scale,model=fit_same_budget_pair_ranker(f,labels)
    scene={"proposals":[{"features":f[0].tolist()},
                        {"features":f[1].tolist()}]}
    score=pair_rank_scores(scale,model,scene)
    assert len(score)==2 and score[0]>score[1]

def test_reject_invalid_training_pairs():
    x=np.random.default_rng(4).normal(size=(20,12))
    with pytest.raises(ValueError):
        fit_same_budget_pair_ranker(x,[1,0,0,1]*5)
    with pytest.raises(ValueError):
        fit_same_budget_pair_ranker(x[:5],[1,0,1,0,1])
    with pytest.raises(ValueError):
        fit_same_budget_pair_ranker(np.full((20,12),np.nan),np.tile([1,0],10))

def test_source_block_bootstrap_and_same_gap_keys():
    rows=[]
    for sid in range(40,100):
        for gap in range(2):
            idx=(sid-40)*2+gap
            for method in ("matched_classification","paired_difference_ranking"):
                rows.append({"section":sid,"gap_id":f"{sid}:{gap}","world":"LINK",
                    "method":method,"true_candidate_available":int(idx<65),
                    "top1_correct":int(idx<(36 if method=="matched_classification" else 40))})
    outcome=evaluate_paired(rows,replicates=200)
    assert outcome["matched_logistic_top1"]==36
    assert outcome["matched_pairwise_ranker_top1"]==40
    assert outcome["only_pair_ranker_correct"]==4
    assert outcome["only_classifier_correct"]==0
    assert outcome["paired_ranker_minus_classifier_top1_fraction"]==pytest.approx(4/120)
    assert evaluate_paired(rows,replicates=200)==outcome
    with pytest.raises(ValueError):
        evaluate_paired(rows[:-1],replicates=100)
