"""Train V53 source-block OOF true-path gate for frozen V52 reconstruction.

Exploratory Thebe TRAIN1 -> already inspected REL-CAL. NO TEST7 read.
The gate is fitted using only source-disjoint TRAIN1 OOF predictions from
retrained ranker and V52 decoder. Original V50 model/ranker/candidates are
kept as controls, using same 120 LINK+120 synthetic NO_LINK worlds.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def run_v53_comparison(cal, scaler, ranker, pathnet, v50_evidence, out_dir, *, augmented_train):
    from scripts.tver_v53_path_accept_gate import (
        fit_path_success_gate, score_source_path_worlds, audit_path_gate)
    from scripts.tver_v52_thebe_comparison import evaluate_paired_cal

    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    # Source-fold OOF training reads TRAIN1 only; it must be finished before
    # any CAL labels are passed to a scored gate. No CAL threshold fitting.
    path_gate,train_receipt=fit_path_success_gate(
        augmented_train,out/"V53_SOURCE_BLOCK_OOF_TRAIN")
    rows=score_source_path_worlds(cal,scaler,ranker,pathnet,path_gate)
    gate_evidence=audit_path_gate(rows)
    # V52 raw decoder scores for matched actual paths, independent of gate.
    raw=evaluate_paired_cal(cal,scaler,ranker,pathnet,v50_evidence,out)
    if (gate_evidence["correct_hidden_paths_from_top1"] != raw["n_v52_correct_paths"]
            or raw["n_v52_correct_paths"] != 33):
        raise RuntimeError("V53 selective gate changed frozen V52 paths")
    report={
        "status":"V53_PATH_AWARE_SELECTIVE_CURVATURE_GATE_POSTHOC_REAL_THEBE_NO_TEST7",
        "source_block_fivefold_TRAIN_only":True,
        "raw_curvature_paths_reproduced":33,
        "matched_legacy_v50_success_paths":18,
        "fixed_eleven_thresholds_same_as_V51":True,
        "synthetic_NO_LINK_not_natural_geological_identity":True,
        "cal_previously_inspected_not_independent":True,
        "no_geological_field_independence_claim":True,
        "TEST7_never_opened":True,
        "source_block_train_receipt":train_receipt,
        "source_CI_conditional_CAL_audit":gate_evidence,
    }
    (out/"V53_PATH_GATE_RESULT.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    with (out/"V53_PATH_GATE_CASES.csv").open("w",encoding="utf8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print("V53_PATH_SUCCESS_GATE_RESULT",json.dumps({
        "status":report["status"],
        "OOF_train_successful_paths":train_receipt["positive_successful_OOF_paths"],
        "n_correct_raw_V52":raw["n_v52_correct_paths"],
        "n_correct_raw_V50":raw["n_original_v50_correct_paths"],
        "best_accepted_correct_paths_conditional_CP95_10pct":gate_evidence["best_accepted_correct_paths_conditional_CP95_10pct"],
        "best_accepted_correct_paths_at_zero_observed_wrong_sections_posthoc":gate_evidence["best_accepted_correct_paths_at_zero_observed_wrong_sections_posthoc"],
        "best_accepted_correct_paths_both_empirical_errors_le10pct_posthoc":gate_evidence["best_accepted_correct_paths_both_empirical_errors_le10pct_posthoc"],
    },indent=2),flush=True)
    return report


def main():
    from scripts.tver_v50_path_net_experiment import execute as v50_execute
    parser=argparse.ArgumentParser()
    for option in ("mmap-root","frontends-dir","v43-receipt-dir","v45-receipt-dir","out-dir"):
        parser.add_argument("--"+option,type=Path,required=True)
    args=parser.parse_args()
    v50_execute(args.mmap_root,args.frontends_dir,
        args.v43_receipt_dir,args.v45_receipt_dir,args.out_dir,
        comparison_hook=run_v53_comparison)


if __name__=='__main__':
    main()