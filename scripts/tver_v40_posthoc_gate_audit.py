"""Posthoc risk-coverage autopsy of immutable V40 real Thebe CASE_ROWS only.

No source images, no TEST7. Never interpret selected CAL thresholds as
independently controlled geological risk or a deployable policy.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

METHODS=("matched_classification","paired_difference_ranking")


def audit(receipt,rows):
    if (receipt.get("status")!="V40_REAL_THEBE_PAIRED_RANKING_EXPLORATORY_COMPLETE_TEST7_UNOPENED"
        or receipt.get("Thebe_TEST7_unopened") is not True):
        raise ValueError("Wrong or uncertified original V40 receipt")
    m=receipt["metrics"]
    if (m["true_candidate_available"]!=65 or m["matched_logistic_top1"]!=36
        or m["matched_pairwise_ranker_top1"]!=41):
        raise ValueError("Baseline result changed")
    data={}
    for name in METHODS:
        cases=[r for r in rows if r["method"]==name]
        pos=[r for r in cases if r["world"]=="LINK"]
        neg=[r for r in cases if r["world"]=="NO_LINK"]
        if (len(pos)!=120 or len(neg)!=120 or
            len({int(r["section"]) for r in pos})!=60 or
            sum(int(r["true_candidate_available"]) for r in pos)!=65):
            raise ValueError("Source or candidate cohort mismatch")
        seen={(r["section"],r["gap_id"],r["world"]) for r in cases}
        if len(seen)!=240:
            raise ValueError("Repeated or missing source gap")
        scored=[]
        for r in cases:
            t=r["top1_score"]
            if t in ("",None):
                continue
            v=float(t)
            if not math.isfinite(v) or not 0<=v<=1:
                raise ValueError("Invalid top1 score")
            scored.append((v,r["world"],int(r["section"]),int(r["top1_correct"])))
        thresholds=sorted({p[0] for p in scored},reverse=True)
        best_zero=best_emp=0
        for t in thresholds:
            admitted=[r for r in scored if r[0]>=t]
            good=sum(r[1]=="LINK" and r[3]==1 for r in admitted)
            bad_link={r[2] for r in admitted if r[1]=="LINK" and r[3]==0}
            bad_no={r[2] for r in admitted if r[1]=="NO_LINK"}
            if not bad_link and not bad_no:
                best_zero=max(best_zero,good)
            if len(bad_link)<=6 and len(bad_no)<=6:
                best_emp=max(best_emp,good)
        good_score=[r[0] for r in scored if r[1]=="LINK" and r[3]]
        bad_score=[r[0] for r in scored if r[1]=="NO_LINK" or
                   (r[1]=="LINK" and not r[3])]
        data[name]={
            "top1_correct_before_any_acceptance":sum(int(r["top1_correct"]) for r in pos),
            "unique_observed_scores":len(thresholds),
            "max_correct_LINK_score":max(good_score,default=None),
            "max_wrong_LINK_or_synthetic_NOLINK_score":max(bad_score,default=None),
            "max_correct_at_zero_observed_bad_sections":best_zero,
            "max_correct_at_both_empirical_section_errors_le_10pct":best_emp,
            "thresholds_selected_POSTHOC":True,
        }
    return {"status":"V40_REAL_THEBE_SCORE_GATE_POSTHOC_DIAGNOSTIC_COMPLETE",
            "original_CAL_already_inspected":True,
            "original_TEST7_unopened":True,
            "geological_field_independence_proven":False,
            "natural_fault_instance_negative_labels":False,
            "conditional_risk_guarantee":False,"methods":data}


def run(source,out):
    source=Path(source);out=Path(out)
    expected={"V40_RESULT.json","V40_CASE_ROWS.csv","V40_CODE_SHA.txt"}
    paths={}
    for item in expected:
        matches=list(source.rglob(item))
        if len(matches)!=1:
            raise ValueError("Missing or ambiguous immutable V40 evidence: "+item)
        paths[item]=matches[0]
    report=json.loads(paths["V40_RESULT.json"].read_text(encoding="utf8"))
    with paths["V40_CASE_ROWS.csv"].open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    out.mkdir(parents=True,exist_ok=True)
    ans=audit(report,rows)
    ans["original_V40_commit"]=paths["V40_CODE_SHA.txt"].read_text().strip()
    (out/"V40_POSTHOC_SCORE_GATE_AUDIT.json").write_text(json.dumps(ans,indent=2)+"\n")
    print(json.dumps(ans,indent=2),flush=True)
    return ans

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
