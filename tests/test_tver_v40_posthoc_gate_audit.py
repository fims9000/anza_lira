"""No-source tests for V40 posthoc score-gate audit."""
import pytest
from scripts.tver_v40_posthoc_gate_audit import audit


def fixture():
    receipt={"status":"V40_REAL_THEBE_PAIRED_RANKING_EXPLORATORY_COMPLETE_TEST7_UNOPENED",
             "Thebe_TEST7_unopened":True,
             "metrics":{"true_candidate_available":65,
                        "matched_logistic_top1":36,"matched_pairwise_ranker_top1":41}}
    rows=[]
    for method in ("matched_classification","paired_difference_ranking"):
        for section in range(60):
            for i in range(2):
                gap=section*2+i
                for world in ("LINK","NO_LINK"):
                    good=world=="LINK" and gap<(36 if method=="matched_classification" else 41)
                    # Deliberate invalid confidence separator: wrong cases score highest.
                    score=.99 if world=="NO_LINK" else (.95 if good else .97)
                    rows.append({"method":method,"section":str(section),"gap_id":str(gap),
                                 "world":world,"top1_score":str(score),
                                 "top1_correct":str(int(good)),
                                 "true_candidate_available":str(int(world=="LINK" and gap<65))})
    return receipt,rows


def test_zero_and_empirical_mismatch_with_nonzero_top1():
    receipt,rows=fixture()
    x=audit(receipt,rows)
    assert x["methods"]["matched_classification"]["top1_correct_before_any_acceptance"]==36
    assert x["methods"]["paired_difference_ranking"]["top1_correct_before_any_acceptance"]==41
    assert x["methods"]["paired_difference_ranking"]["max_correct_at_zero_observed_bad_sections"]==0
    assert x["methods"]["paired_difference_ranking"]["max_correct_at_both_empirical_section_errors_le_10pct"]==0


def test_stop_wrong_source_and_nonfinite_scores():
    receipt,rows=fixture()
    receipt["Thebe_TEST7_unopened"]=False
    with pytest.raises(ValueError):
        audit(receipt,rows)
    receipt["Thebe_TEST7_unopened"]=True
    rows[0]["top1_score"]="nan"
    with pytest.raises(ValueError,match="score"):
        audit(receipt,rows)
    rows[0]["top1_score"]=".95"
    with pytest.raises(ValueError,match="cohort"):
        audit(receipt,rows[:-1])
