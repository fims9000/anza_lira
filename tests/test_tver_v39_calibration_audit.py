"""Pure V39 mathematical/provenance tests; no Thebe source or TEST7."""
import copy
import math
import pytest
from scripts.tver_v39_calibration_audit import upper_cp, audit, paired_ranking_audit, METHODS, MODEL_SHA

def frozen_sample():
    scans={}
    for i in range(21):
        t=str(round(i*.05,2))
        scans[t]={"correctly_accepted_positive_gap_cases":20,
                  "total_accepted_positive_gap_cases":30,
                  "wrong_accepted_LINK_source_sections":7,
                  "accepted_candidate_absent_NOLINK_source_sections":7,
                  "wrong_link_CP_upper_95":upper_cp(7,60),
                  "nolink_CP_upper_95":upper_cp(7,60)}
    scans["1.01"]={"correctly_accepted_positive_gap_cases":0,
                   "total_accepted_positive_gap_cases":0,
                   "wrong_accepted_LINK_source_sections":0,
                   "accepted_candidate_absent_NOLINK_source_sections":0,
                   "wrong_link_CP_upper_95":upper_cp(0,60),
                   "nolink_CP_upper_95":upper_cp(0,60)}
    report={"status":"V34_THEBE_ANZA_NATIVE_P0_AND_LOGISTIC_SECTION_CAL_COMPLETE_TEST_UNOPENED",
        "ANZA_exact_checkpoint_SHA256":MODEL_SHA,
        "ANZA_exact_topology_threshold":.65,
        "external_test7_images_labels_model_scores_NEVER_OPENED":True,
        "RELATION_CAL_frozen_policies":{
          m:{"source_section_count":60,"accept_threshold":1.01,"scan_by_threshold":copy.deepcopy(scans)}
          for m in METHODS}}
    rows=[]
    for method in METHODS:
        for section in range(40,100):
            for gap in range(2):
                for world in ("LINK","NO_LINK"):
                    i=(section-40)*2+gap
                    rows.append({"model":method,"section":str(section),"artificial_gap_id":f"thebe:{section}:{gap}","world":world,
                                 "calibration_accept_threshold":"1.01","accepted_ANY":"False",
                                 "at_least_one_true_candidate":str(world=="LINK" and i<65),
                                 "top1_correct_BEFORE_risk_gate":str(world=="LINK" and i<36)})
    return report,rows

def test_zero_error_bound_and_minimum_n():
    assert upper_cp(0,60)==pytest.approx(.04870291331009746)
    assert upper_cp(0,60,.05/44)==pytest.approx(.10684817741001618)
    assert upper_cp(1,60)>upper_cp(0,60)
    assert math.ceil(math.log(.05/44)/math.log(.9))==65

def test_all_methods_and_selected_safety_refusal():
    report,rows=frozen_sample()
    z=audit(report,rows)
    assert z["Thebe_TEST7_opened"] is False
    for model in METHODS:
        x=z["method_results"][model]
        assert x["candidate_available_at_065"]==65
        assert x["top1_correct_before_refusal"]==36
        assert x["accepted_at_frozen_threshold"]==0
        assert x["best_correct_gaps_simultaneous_gate"]==0

def test_provenance_and_source_fail_closed():
    report,rows=frozen_sample()
    for key,wrong in [("ANZA_exact_checkpoint_SHA256","wrong"),
                      ("external_test7_images_labels_model_scores_NEVER_OPENED",False),
                      ("ANZA_exact_topology_threshold",.5)]:
        changed=copy.deepcopy(report);changed[key]=wrong
        with pytest.raises(ValueError):
            audit(changed,rows)
    with pytest.raises(ValueError):
        audit(report,rows[:-1])

def test_cp_receipt_corruption_detected():
    report,rows=frozen_sample()
    report["RELATION_CAL_frozen_policies"]["geometry"]["scan_by_threshold"]["0.1"]["wrong_link_CP_upper_95"]+=.1
    with pytest.raises(ValueError,match="CP receipt mismatch"):
        audit(report,rows)

def test_paired_ranking_complementarity_and_reproducible_section_bootstrap():
    _,rows=frozen_sample()
    result=paired_ranking_audit(rows,bootstrap_replicates=500)
    x=result["comparisons"]["matched_logistic_vs_p0_seed11"]
    assert x["both_correct"]==36
    assert x["only_matched_logistic_correct"]==0
    assert x["only_comparator_correct"]==0
    altered=copy.deepcopy(rows)
    # Two P0-only positive cases that matched logistic misses.
    changed=0
    for row in altered:
        if row["model"]=="p0_seed11" and row["world"]=="LINK" and row["section"]=="60" and changed<2:
            row["top1_correct_BEFORE_risk_gate"]="True"
            changed+=1
    actual=paired_ranking_audit(altered,bootstrap_replicates=500)
    y=actual["comparisons"]["matched_logistic_vs_p0_seed11"]
    assert y["only_comparator_correct"]==2
    assert y["oracle_union_top1_upper_bound_not_deployable"]==38
    assert paired_ranking_audit(altered,bootstrap_replicates=500)==actual
    altered=[r for r in altered if not (r["model"]=="p0_seed11" and r["world"]=="LINK" and r["artificial_gap_id"]=="thebe:60:0")]
    with pytest.raises(ValueError,match="identical gaps"):
        paired_ranking_audit(altered,bootstrap_replicates=200)
