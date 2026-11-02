#!/usr/bin/env python3
"""Aggregate five CT28 development-validation CCTA mask repairs, no test access.

Patient-macro paired Dice/clDice and false inter-tree GT bridge evidence.
Missing and failed subjects count against completeness; never silently omitted.
Intervals are descriptive and NOT independent after K10/15 val model selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from ccta_3d_val_freeze import verify_validation_freeze,sha
from ccta_3d_val_pilot import DEFAULT_OUTPUT,run_validation_pilot
from ccta_3d_train_gap_simulator import VAL_CT28_IDS


def read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_validation(
    *,
    root:Path,
    protocol:Path,
    patient_ids:list[int]|None=None,
    bootstrap:int=10000,
    seed:int=20261102,
)->tuple[dict,list[dict]]:
    val_ids=sorted(VAL_CT28_IDS) if patient_ids is None else patient_ids
    if not val_ids or len(val_ids)!=len(set(val_ids)) or not set(val_ids)<=VAL_CT28_IDS:
        raise ValueError("Only five CT28 development-validation patients allowed")
    if bootstrap<100:
        raise ValueError("At least 100 descriptive bootstrap draws required")
    freeze=verify_validation_freeze(protocol)
    expected_sha=sha(protocol)
    rows=[]
    for patient in val_ids:
        folder=root/f"patient_{patient}"
        manifest=folder/"pilot_manifest.json"
        fail=folder/"pilot_failure.json"
        record={"patient_id":patient}
        if fail.exists():
            error=read(fail)
            record.update(
                status="failed",
                error_type=error.get("error_type","unknown"),
                error=str(error.get("error",""))[:300],
            )
        elif not manifest.exists():
            record.update(status="missing")
        else:
            run=read(manifest)
            if (
                run.get("status")!="complete_frozen_ct28_development_validation_pilot"
                or run.get("patient_id")!=patient
                or run.get("split")!="val"
                or run.get("protocol_sha256")!=expected_sha
                or run.get("held_out_test_accessed") is not False
                or run.get("ground_truth_used_for_inference") is not False
                or run.get("frozen_selected_model_sha256")!=freeze["selected_K10_SHA256"]
            ):
                raise ValueError(f"Invalid/mixed validation freeze receipt for patient {patient}")
            metric_file=folder/"posthoc_metrics.json"
            decision_file=folder/"decisions.json"
            if not metric_file.is_file() or not decision_file.is_file():
                raise ValueError(f"Missing full-mask inference evidence for {patient}")
            metrics=read(metric_file)
            decisions=read(decision_file)
            if metrics.get("patient_id")!=str(patient) or decisions.get("patient_label")!=str(patient):
                raise ValueError("Mixed patient in validation prediction or metric")
            if metrics.get("before_sha256")!=sha(folder/"broken_mask.nii.gz"):
                raise ValueError("Broken mask provenance mismatch")
            if metrics.get("after_sha256")!=sha(folder/"repaired.nii.gz"):
                raise ValueError("Repaired mask provenance mismatch")
            required={
                "dice_before","dice_after","dice_delta",
                "cldice_before","cldice_after","cldice_delta",
                "reference_unsupported_added_voxels",
                "gt_inconsistent_new_component_pairs",
                "gt_unassigned_new_component_pairs",
                "added_voxels",
            }
            if not required<=set(metrics):
                raise ValueError("New posthoc GT topology bridge metrics missing")
            vals=np.array([metrics[k] for k in required],dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError("Nonfinite validation segmentation quality measurement")
            if int(metrics["added_voxels"])!=int(decisions["added_voxels"]):
                raise ValueError("Added voxel count mismatches original inference")
            record.update(
                status="complete",
                scenes=int(decisions["scene_count"]),
                repairs=int(decisions["accepted_decisions"]),
                added_voxels=int(metrics["added_voxels"]),
                gt_unsupported_added_voxels=int(metrics["reference_unsupported_added_voxels"]),
                gt_inconsistent_component_pairs=int(metrics["gt_inconsistent_new_component_pairs"]),
                gt_unknown_component_pairs=int(metrics["gt_unassigned_new_component_pairs"]),
                dice_before=float(metrics["dice_before"]),
                dice_after=float(metrics["dice_after"]),
                dice_delta=float(metrics["dice_delta"]),
                cldice_before=float(metrics["cldice_before"]),
                cldice_after=float(metrics["cldice_after"]),
                cldice_delta=float(metrics["cldice_delta"]),
            )
        rows.append(record)
    complete=[record for record in rows if record["status"]=="complete"]
    rng=random.Random(seed)
    comparison={}
    for name in ("dice_delta","cldice_delta"):
        if not complete:
            comparison[name]=None
            continue
        vals=[record[name] for record in complete]
        draws=sorted(
            sum(rng.choice(vals) for _ in vals)/len(vals)
            for _ in range(bootstrap)
        )
        comparison[name]={
            "mean_patient_delta":float(np.mean(vals)),
            "descriptive_patient_bootstrap_95pct":[
                draws[int(.025*(bootstrap-1))],
                draws[int(.975*(bootstrap-1))],
            ],
        }
    status=(
        "complete_five_subject_development_validation"
        if len(val_ids)==5 and len(complete)==5
        else "partial_or_subset_development_validation"
    )
    report={
        "status":status,
        "scope":"CT28 five historical development-validation subjects, not held-out",
        "frozen_protocol_sha256":expected_sha,
        "patients_requested":val_ids,
        "patients_completed":len(complete),
        "patients_failed":sum(row["status"]=="failed" for row in rows),
        "patients_missing":sum(row["status"]=="missing" for row in rows),
        "total_repairs":sum(row["repairs"] for row in complete),
        "total_added_voxels":sum(row["added_voxels"] for row in complete),
        "total_gt_unsupported_added_voxels":sum(row["gt_unsupported_added_voxels"] for row in complete),
        "total_gt_inconsistent_component_pairs":sum(row["gt_inconsistent_component_pairs"] for row in complete),
        "total_gt_unknown_component_pairs":sum(row["gt_unknown_component_pairs"] for row in complete),
        "patient_macro":comparison,
        "bootstrap_draws":bootstrap,
        "bootstrap_seed":seed,
        "held_out_test_accessed":False,
        "limitations":[
            "These 5 subjects were used to select K10/15 at scene level; result is not confirmatory.",
            "Controlled gap corruption of reference masks is not natural segmentation failure.",
            "False GT cross-component merge count does not detect incorrect branches within one GT component.",
            "Only a predeclared, one-time independent test can support final generalization claims.",
        ],
    }
    return report,rows


def write_validation_summary(root:Path,report:dict,rows:list[dict])->None:
    paths=[
        root/"validation_cohort_summary.json",
        root/"validation_cohort_patients.csv",
        root/"validation_cohort_results.md",
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("Validation summary is immutable; use a new root")
    root.mkdir(parents=True,exist_ok=True)
    paths[0].write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    columns=sorted({key for record in rows for key in record})
    with paths[1].open("w",newline="",encoding="utf-8") as source:
        writer=csv.DictWriter(source,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result=[
        "# Frozen CT28 CCTA 3D development-validation, five patients",
        "",
        f"Status: **{report['status']}**; complete {report['patients_completed']}/{len(report['patients_requested'])}.",
        "",
        "| Patient | Status | Repairs | Dice before | Dice after | clDice before | clDice after | GT-inconsistent merged component pairs |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def field(name):
            val=row.get(name)
            return "—" if val is None else f"{val:.4f}" if isinstance(val,float) else str(val)
        result.append(
            f"| {row['patient_id']} | {row['status']} | {field('repairs')} | "
            f"{field('dice_before')} | {field('dice_after')} | {field('cldice_before')} | "
            f"{field('cldice_after')} | {field('gt_inconsistent_component_pairs')} |"
        )
    result+=["","## Patient-macro paired changes", ""]
    for name,stats in report["patient_macro"].items():
        if stats is None:
            result.append(f"- {name}: NOT MEASURED")
        else:
            low,high=stats["descriptive_patient_bootstrap_95pct"]
            result.append(f"- {name}: {stats['mean_patient_delta']:+.5f}, descriptive cluster 95% [{low:+.5f},{high:+.5f}]")
    result+=["","## Limitations",""]+[f"- {item}" for item in report["limitations"]]
    paths[2].write_text("\n".join(result)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--patients",type=int,nargs="+",default=sorted(VAL_CT28_IDS))
    parser.add_argument("--bootstrap",type=int,default=10000)
    args=parser.parse_args()
    if not set(args.patients)<=VAL_CT28_IDS or len(set(args.patients))!=len(args.patients):
        raise ValueError("Requested held-out test or duplicated validation subject")
    # Explicit verification before visiting even the first validation image.
    verify_validation_freeze(args.freeze)
    if args.run:
        args.root.mkdir(parents=True,exist_ok=True)
        for patient in args.patients:
            target=args.root/f"patient_{patient}"
            if target.exists():
                print(f"Preserving immutable validation evidence for {patient}",flush=True)
                continue
            try:
                run_validation_pilot(
                    patient=patient,freeze_path=args.freeze,
                    output_dir=target,
                )
                print(f"Completed frozen development-val patient {patient}",flush=True)
            except Exception as exc:
                print(f"FAILED validation patient {patient}: {exc}",flush=True)
    summary,rows=summarize_validation(
        root=args.root,protocol=args.freeze,
        patient_ids=args.patients,bootstrap=args.bootstrap,
    )
    write_validation_summary(args.root,summary,rows)
    print(json.dumps({
        key:summary[key] for key in (
            "status","patients_completed","patients_failed",
            "patients_missing","total_gt_inconsistent_component_pairs","patient_macro"
        )
    },indent=2))
    if args.run and summary["status"]!="complete_five_subject_development_validation":
        raise RuntimeError("Development-validation incomplete: inspect per-patient failure evidence")


if __name__=="__main__":
    main()
