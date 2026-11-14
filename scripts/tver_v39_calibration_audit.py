"""V39: independent audit of the existing V34 Thebe relation CAL receipts.

This reads saved V34 aggregate metrics ONLY: no Thebe seismic arrays, no new
training, no labels from an untouched TEST. Exploratory/posthoc, not confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

SOURCE_COMMIT = "8f2dd0fbb322fb295976513dd98ada342d255991"
MODEL_SHA = "ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e"
METHODS = ("geometry", "full_train_logistic", "matched_logistic", "p0_seed11", "p0_seed23")
ALPHA = .05
RISK_TARGET = .10


def upper_cp(k, n, alpha=ALPHA):
    """Exact one-sided binomial upper bound, conditional on independence."""
    if n < 1 or k < 0 or k > n or not 0 < alpha < 1:
        raise ValueError("invalid binomial arguments")
    if k == n:
        return 1.
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    lo, hi = 0., 1.
    for _ in range(70):
        q = (lo + hi) / 2
        p = (1 - q) ** n
        tail = p
        for j in range(k):
            p *= (n - j) * q / ((j + 1) * (1 - q))
            tail += p
        if tail > alpha:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2


def parse_bool(x):
    if x in ("True", "true", "1"):
        return True
    if x in ("False", "false", "0"):
        return False
    raise ValueError("invalid boolean encoding")


def audit(receipt, case_rows):
    if receipt.get("status") != "V34_THEBE_ANZA_NATIVE_P0_AND_LOGISTIC_SECTION_CAL_COMPLETE_TEST_UNOPENED":
        raise ValueError("original V34 experiment incomplete")
    if receipt.get("ANZA_exact_checkpoint_SHA256") != MODEL_SHA:
        raise ValueError("frozen model changed")
    if receipt.get("external_test7_images_labels_model_scores_NEVER_OPENED") is not True:
        raise ValueError("TEST7 embargo violated")
    if receipt.get("ANZA_exact_topology_threshold") != .65:
        raise ValueError("original segmenter threshold changed")
    policies = receipt.get("RELATION_CAL_frozen_policies", {})
    if set(policies) != set(METHODS):
        raise ValueError("unexpected baseline model set")
    out = {}
    for method in METHODS:
        rows = [r for r in case_rows if r["model"] == method]
        positive = [r for r in rows if r["world"] == "LINK"]
        negative = [r for r in rows if r["world"] == "NO_LINK"]
        if len(positive) != 120 or len(negative) != 120:
            raise ValueError("not original 120+120 controlled gap worlds")
        sections = {int(r["section"]) for r in positive}
        if len(sections) != 60 or sections != {int(r["section"]) for r in negative}:
            raise ValueError("original source-section cohort changed")
        for cohort in (positive, negative):
            if set(Counter(int(r["section"]) for r in cohort).values()) != {2}:
                raise ValueError("not two gaps per source")
        p = policies[method]
        n = int(p["source_section_count"])
        if n != 60:
            raise ValueError("not the original 60 source sections")
        scan = p["scan_by_threshold"]
        if len(scan) != 22 or "1.01" not in scan:
            raise ValueError("unexpected threshold search")
        a_joint = ALPHA / (2 * len(scan))
        curve = []
        for t, raw in scan.items():
            bad_link = int(raw["wrong_accepted_LINK_source_sections"])
            bad_no = int(raw["accepted_candidate_absent_NOLINK_source_sections"])
            plain_link = upper_cp(bad_link, n)
            plain_no = upper_cp(bad_no, n)
            if abs(plain_link - float(raw["wrong_link_CP_upper_95"])) > 1e-8:
                raise ValueError("original LINK CP receipt mismatch")
            if abs(plain_no - float(raw["nolink_CP_upper_95"])) > 1e-8:
                raise ValueError("original NO_LINK CP receipt mismatch")
            bound_link = upper_cp(bad_link, n, a_joint)
            bound_no = upper_cp(bad_no, n, a_joint)
            curve.append({
                "threshold": float(t),
                "correctly_accepted_positive_gaps": int(raw["correctly_accepted_positive_gap_cases"]),
                "accepted_positive_gaps": int(raw["total_accepted_positive_gap_cases"]),
                "wrong_accepted_LINK_sections": bad_link,
                "false_accepted_synthetic_NOLINK_sections": bad_no,
                "nominal_95_LINK_upper": plain_link, "nominal_95_NOLINK_upper": plain_no,
                "joint_95_LINK_upper": bound_link, "joint_95_NOLINK_upper": bound_no,
                "original_nominal_gate": max(plain_link, plain_no) <= RISK_TARGET,
                "simultaneous_joint_gate": max(bound_link, bound_no) <= RISK_TARGET,
            })
        curve.sort(key=lambda z: z["threshold"])
        chosen = float(p["accept_threshold"])
        if not any(abs(z["threshold"]-chosen) < 1e-9 for z in curve):
            raise ValueError("chosen threshold not in original scan")
        if any(abs(float(r["calibration_accept_threshold"])-chosen)>1e-9 for r in rows):
            raise ValueError("case decisions do not match frozen selection")
        accepted = sum(parse_bool(r["accepted_ANY"]) for r in rows)
        if chosen == 1.01 and accepted:
            raise ValueError("never-link unexpectedly accepted a pair")
        available = sum(parse_bool(r["at_least_one_true_candidate"]) for r in positive)
        top1 = sum(parse_bool(r["top1_correct_BEFORE_risk_gate"]) for r in positive)
        if top1 > available:
            raise ValueError("ranking exceeds candidate recall")
        out[method] = {
            "candidate_available_at_065": available,
            "top1_correct_before_refusal": top1,
            "frozen_decision_threshold": chosen,
            "accepted_at_frozen_threshold": accepted,
            "best_correct_gaps_nominal_gate": max((r["correctly_accepted_positive_gaps"] for r in curve if r["original_nominal_gate"]), default=0),
            "best_correct_gaps_simultaneous_gate": max((r["correctly_accepted_positive_gaps"] for r in curve if r["simultaneous_joint_gate"]), default=0),
            "full_22_threshold_curve": curve,
        }
    min_n = math.ceil(math.log(ALPHA / (22 * 2)) / math.log(1 - RISK_TARGET))
    return {
        "status": "V39_AUDITED_ORIGINAL_V34_CAL_WITHOUT_NEW_TEST",
        "original_V34_commit": SOURCE_COMMIT,
        "frozen_ANZA_SHA256": MODEL_SHA,
        "independent_section_count_assumed_not_proven": 60,
        "within_field_section_independence_verified": False,
        "natural_geological_false_link_ground_truth": False,
        "Thebe_TEST7_opened": False,
        "already_viewed_relation_CAL_posthoc": True,
        "one_sided_confidence": .95,
        "risk_target": RISK_TARGET,
        "familywise_22_thresholds_times_2_error_types": 44,
        "zero_error_nominal_upper": upper_cp(0, 60),
        "zero_error_simultaneous_upper": upper_cp(0, 60, ALPHA / 44),
        "min_independent_sections_for_joint_10pct_zero_error_bound": min_n,
        "method_results": out,
        "scope": "Bonferroni is a sufficient, conservative post-selection correction, not an optimal procedure. Section-level CP assumes independent units; all sections are from one field. No comparison to natural fault identities is established.",
    }


def paired_ranking_audit(case_rows, seed=20261114, bootstrap_replicates=10000):
    """Posthoc paired ranking differences, resampling whole source sections.

    These are descriptive same-field intervals, NOT independent geological
    population confidence guarantees or new held-out validation.
    """
    import random
    positive={}
    for method in METHODS:
        positive[method]={}
        for row in case_rows:
            if row["model"] != method or row["world"] != "LINK":
                continue
            key=(int(row["section"]),row["artificial_gap_id"])
            if key in positive[method]:
                raise ValueError("duplicate paired source/gap")
            positive[method][key]=int(parse_bool(row["top1_correct_BEFORE_risk_gate"]))
    common=set(positive["matched_logistic"])
    if not common or len(common)!=120 or len({s for s,_ in common})!=60:
        raise ValueError("original paired source/gap cohort changed")
    if any(set(values)!=common for values in positive.values()):
        raise ValueError("ranking models are not evaluated on identical gaps")
    sections=sorted({s for s,_ in common})
    rng=random.Random(seed)
    out={}
    for comparator in METHODS:
        if comparator=="matched_logistic":
            continue
        a=positive["matched_logistic"]
        b=positive[comparator]
        both=sum(a[k] and b[k] for k in common)
        only_a=sum(a[k] and not b[k] for k in common)
        only_b=sum(b[k] and not a[k] for k in common)
        neither=len(common)-both-only_a-only_b
        section_differences={
            s:sum(a[k]-b[k] for k in common if k[0]==s)
            for s in sections}
        # Every source section contributes two artificial gaps.
        draws=[]
        for _ in range(bootstrap_replicates):
            gap_difference=sum(section_differences[sections[rng.randrange(len(sections))]]
                               for __ in sections)
            draws.append(gap_difference/len(common))
        draws.sort()
        lower=draws[int(.025*bootstrap_replicates)]
        upper=draws[min(bootstrap_replicates-1,int(.975*bootstrap_replicates))]
        out["matched_logistic_vs_"+comparator]={
            "both_correct":both,
            "only_matched_logistic_correct":only_a,
            "only_comparator_correct":only_b,
            "neither_correct":neither,
            "oracle_union_top1_upper_bound_not_deployable":both+only_a+only_b,
            "matched_minus_comparator_top1_fraction":(only_a-only_b)/120,
            "whole_section_bootstrap_95_descriptive_interval":[lower,upper],
            "source_sections_where_matched_better":sum(x>0 for x in section_differences.values()),
            "source_sections_where_comparator_better":sum(x<0 for x in section_differences.values()),
        }
    return {"status":"V39_POSTHOC_PAIRED_RANKING_AUDIT",
            "independent_field_replication":False,
            "bootstrap_unit":"source section, 2 artificially hidden gaps each",
            "bootstrap_replicates":bootstrap_replicates,
            "seed":seed,"comparisons":out}


def run(input_dir: Path, output_dir: Path):
    paths = {}
    for name, leaf in [("receipt", "v34_THEBE_P0_ANZA_source_CAL_frozen.json"),
                       ("cases", "v34_THEBE_ANZA_P0_CAL_case_rows.csv"),
                       ("commit", "COMMIT_SHA.txt")]:
        matched = list(input_dir.rglob(leaf))
        if len(matched) != 1:
            raise ValueError("V34 immutable artifact missing or ambiguous: " + leaf)
        paths[name] = matched[0]
    if paths["commit"].read_text().strip() != SOURCE_COMMIT:
        raise ValueError("V34 original commit drift")
    source = json.loads(paths["receipt"].read_text(encoding="utf8"))
    with paths["cases"].open(encoding="utf8", newline="") as f:
        cases = list(csv.DictReader(f))
    evidence = audit(source, cases)
    evidence["paired_top1_ranking"] = paired_ranking_audit(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "V39_V34_RISK_AUDIT.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    brief = {k:v for k,v in evidence.items() if k != "method_results"}
    brief["methods"] = {m:{k:v for k,v in record.items() if k != "full_22_threshold_curve"}
                        for m,record in evidence["method_results"].items()}
    (output_dir / "V39_BRIEF.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(brief, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args=ap.parse_args()
    run(args.artifact_dir, args.out_dir)
