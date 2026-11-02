#!/usr/bin/env python3
"""Compare K10 vs 3D CNN vs 3D ANZA on *five frozen CT28 validation masks*.

The exact same CT, damaged segmentation and reference are used for all three.
Outputs are immutable and both neural encoders must be trained on the 17 train
patients only. This is development-validation (previously K10 selected on these
subjects), NOT the blind six-patient Graph-LIRA test or clinical deployment.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ccta_3d_an_cli import run_neural
from ccta_3d_evaluate import evaluate_files
from ccta_3d_train_pilot import SELECTED_PACK,REFERENCE_DIR
from ccta_3d_val_freeze import verify_validation_freeze,sha
from ccta_3d_train_gap_simulator import VAL_CT28_IDS
from ccta_3d_val_pilot import DEFAULT_OUTPUT


ROOT=Path(__file__).resolve().parents[3]
DEFAULT_STUDY=ROOT/"varvara/anza_lira_local_results/ccta_3d_runtime_pilot/anza_vs_cnn_development"
DEFAULT_COMPARE=ROOT/"varvara/anza_lira_local_results/ccta_3d_validation_neural"


def read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_val_pilot(
    *,
    pilot_dir:Path,patient:int,freeze_path:Path,frozen:dict
)->None:
    if patient not in VAL_CT28_IDS:
        raise ValueError("Only five CT28 development-val patients permitted")
    meta=read(pilot_dir/"pilot_manifest.json")
    if (
        meta.get("status")!="complete_frozen_ct28_development_validation_pilot"
        or meta.get("patient_id")!=patient
        or meta.get("split")!="val"
        or meta.get("protocol_sha256")!=sha(freeze_path)
        or meta.get("ground_truth_used_for_inference") is not False
        or meta.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Wrong or incomplete validation mask/CT pilot evidence")
    if meta.get("ct_sha256")!=frozen["expected_validation_CT_SHA256"][str(patient)]:
        raise ValueError("Validation CCTA source SHA differs from freeze")
    if sha(pilot_dir/"ct.nii.gz")!=meta["ct_sha256"]:
        raise ValueError("CT volume changed after validation pilot")
    baseline=read(pilot_dir/"posthoc_metrics.json")
    if baseline.get("patient_id")!=str(patient):
        raise ValueError("Wrong baseline GT evaluation patient")
    if baseline.get("before_sha256")!=sha(pilot_dir/"broken_mask.nii.gz"):
        raise ValueError("Broken predicted mask changed after baseline prediction")


def pair_experiment_paths(study:Path,variant:str)->tuple[Path,Path,Path]:
    if variant not in {"conv3d","anza3d"}:
        raise ValueError("Unknown neural encoder variant")
    return (
        study/f"relation_{variant}",
        study/f"oof_pair_{variant}",
        study/f"oof_junction_{variant}",
    )


def run_neural_val_patient(
    *,
    patient:int,
    variant:str,
    freeze_path:Path,
    pilot_root:Path,
    study_root:Path,
    output_root:Path,
    device:str,
)->dict:
    if patient not in VAL_CT28_IDS:
        raise ValueError("Held-out test/train patient forbidden in val neural comparison")
    frozen=verify_validation_freeze(freeze_path)
    pilot=pilot_root/f"patient_{patient}"
    check_val_pilot(
        pilot_dir=pilot,patient=patient,
        freeze_path=freeze_path,frozen=frozen,
    )
    relation,pair,junction=pair_experiment_paths(study_root,variant)
    output=output_root/f"patient_{patient}"/variant
    if output.exists():
        raise FileExistsError("One-time val neural result already exists; never overwrite")
    output.mkdir(parents=True,exist_ok=False)
    # A mid-run error deliberately retains partial artifacts and a failure JSON.
    try:
        result=run_neural(
            ct=pilot/"ct.nii.gz",
            mask=pilot/"broken_mask.nii.gz",
            patient_id=str(patient),
            base_k10=SELECTED_PACK,
            relation_dir=relation,pair_oof=pair,junction_oof=junction,
            variant=variant,
            output_mask=output/"repaired.nii.gz",
            output_additions=output/"added_voxels.nii.gz",
            report=output/"decisions.json",
            device=device,
            radius_mm=frozen["settings"]["repair_radius_mm"],
            max_new_voxels=frozen["settings"]["max_new_voxels_per_decision"],
            strict_count_support=frozen["settings"]["strict_count_support"],
        )
        stats=evaluate_files(
            patient_id=str(patient),before=pilot/"broken_mask.nii.gz",
            after=output/"repaired.nii.gz",
            reference=REFERENCE_DIR/f"{patient}.coronary.nii.gz",
            output=output/"posthoc_metrics.json",
        )
        if (
            stats["before_sha256"]!=read(pilot/"posthoc_metrics.json")["before_sha256"]
            or stats["reference_sha256"]!=read(pilot/"posthoc_metrics.json")["reference_sha256"]
        ):
            raise ValueError("Neural val treatment and K10 baseline used different masks or GT")
        receipt={
            "status":"complete_exploratory_neural_CT28_development_validation",
            "patient_id":patient,"variant":variant,
            "freeze_sha256":sha(freeze_path),
            "model_pack_sha256":frozen["selected_K10_SHA256"],
            "source_broken_mask_sha256":stats["before_sha256"],
            "source_reference_sha256":stats["reference_sha256"],
            "repaired_mask_sha256":stats["after_sha256"],
            "accepted_repairs":result["accepted_decisions"],
            "added_voxels":result["added_voxels"],
            "dice_delta":stats["dice_delta"],
            "cldice_delta":stats["cldice_delta"],
            "gt_inconsistent_new_component_pairs":stats["gt_inconsistent_new_component_pairs"],
            "held_out_test_accessed":False,
            "warning":"Relation model trained on clean OOF only; not augmentation-matched to original K10/15.",
        }
        (output/"result_manifest.json").write_text(
            json.dumps(receipt,ensure_ascii=False,indent=2)+"\n"
        )
        return receipt
    except Exception as exc:
        (output/"failure.json").write_text(
            json.dumps({
                "status":"failed_exploratory_neural_CT28_dev_val",
                "patient_id":patient,
                "variant":variant,
                "error_type":type(exc).__name__,
                "error":str(exc),
            },ensure_ascii=False,indent=2)+"\n"
        )
        raise


def summarize_matched_validation(
    *,
    freeze_path:Path,
    pilot_root:Path,
    output_root:Path,
    bootstrap:int=10000,
    seed:int=20261102,
)->tuple[dict,list[dict]]:
    if bootstrap<100:
        raise ValueError("At least 100 patient-resampled bootstrap draws required")
    freeze=verify_validation_freeze(freeze_path)
    rows=[]
    for patient in sorted(VAL_CT28_IDS):
        pilot=pilot_root/f"patient_{patient}"
        baseline_metrics=pilot/"posthoc_metrics.json"
        if not baseline_metrics.exists():
            rows.append({"patient_id":patient,"status":"missing_K10_baseline"})
            continue
        check_val_pilot(
            pilot_dir=pilot,patient=patient,
            freeze_path=freeze_path,frozen=freeze,
        )
        k10=read(baseline_metrics)
        metrics={"K10":k10}
        status="complete"
        for variant in ("conv3d","anza3d"):
            report_path=output_root/f"patient_{patient}"/variant/"result_manifest.json"
            fail_path=report_path.parent/"failure.json"
            if fail_path.exists():
                status="failed_"+variant
                break
            if not report_path.is_file():
                status="missing_"+variant
                break
            run=read(report_path)
            if (
                run.get("status")!="complete_exploratory_neural_CT28_development_validation"
                or run.get("patient_id")!=patient
                or run.get("variant")!=variant
                or run.get("freeze_sha256")!=sha(freeze_path)
            ):
                raise ValueError("Neural CCTA validation receipt or freeze mismatch")
            evaluation=read(report_path.parent/"posthoc_metrics.json")
            if (
                evaluation.get("before_sha256")!=k10["before_sha256"]
                or evaluation.get("reference_sha256")!=k10["reference_sha256"]
                or evaluation.get("after_sha256")!=sha(report_path.parent/"repaired.nii.gz")
            ):
                raise ValueError("Neural comparison uses differing CT/GT/mask provenance")
            if not all(key in evaluation for key in (
                "dice_delta","cldice_delta","gt_inconsistent_new_component_pairs",
                "reference_unsupported_added_voxels"
            )):
                raise ValueError("Missing posthoc segmentation quality or false-bridge measurements")
            metrics[variant]=evaluation
        row={"patient_id":patient,"status":status}
        if status=="complete":
            for model,metric in metrics.items():
                row[model+"_dice_delta"]=float(metric["dice_delta"])
                row[model+"_cldice_delta"]=float(metric["cldice_delta"])
                row[model+"_gt_false_bridge_pairs"]=int(metric["gt_inconsistent_new_component_pairs"])
                row[model+"_unsupported_added_voxels"]=int(metric["reference_unsupported_added_voxels"])
            row["ANZA_minus_CNN_cldice_delta"]=row["anza3d_cldice_delta"]-row["conv3d_cldice_delta"]
            row["ANZA_minus_CNN_dice_delta"]=row["anza3d_dice_delta"]-row["conv3d_dice_delta"]
        rows.append(row)

    complete=[row for row in rows if row["status"]=="complete"]
    rng=random.Random(seed)
    paired={}
    for metric in ("ANZA_minus_CNN_dice_delta","ANZA_minus_CNN_cldice_delta"):
        if not complete:
            paired[metric]=None
            continue
        values=[item[metric] for item in complete]
        boots=sorted(
            sum(rng.choice(values) for _ in values)/len(values)
            for _ in range(bootstrap)
        )
        paired[metric]={
            "mean_paired_patient_delta":float(np.mean(values)),
            "descriptive_patient_bootstrap_95pct":[
                boots[int(.025*(bootstrap-1))],
                boots[int(.975*(bootstrap-1))],
            ],
        }
    report={
        "status":(
            "complete_five_patient_matched_CCTA_dev_val"
            if len(complete)==5 else
            "partial_CCTA_development_validation_neural_comparison"
        ),
        "patient_count":5,
        "patient_completed":len(complete),
        "patient_failed_or_missing":5-len(complete),
        "protocol_sha256":sha(freeze_path),
        "paired_ANZA_minus_CNN":paired,
        "held_out_test_accessed":False,
        "limitations":[
            "K10/15 selected on the same five development-validation patients; not independent.",
            "ANN relation heads are clean-OOF trained; K10 used 15 train augmentations.",
            "Only controlled artificial GT gaps, not natural predicted segmentation errors.",
            "A 5-patient bootstrap interval is descriptive, not a confirmatory CI.",
            "Different tree branches within one GT component may still be incorrectly bridged.",
        ],
    }
    return report,rows


def write_matched(root:Path,report:dict,rows:list[dict]):
    destination=root/"matched_val_comparison.json"
    table=root/"matched_val_comparison.md"
    if destination.exists() or table.exists():
        raise FileExistsError("Cannot overwrite immutable matched validation report")
    root.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({
        **report,"per_patient":rows
    },ensure_ascii=False,indent=2)+"\n")
    lines=[
        "# CT28 CCTA development-validation: K10 vs CNN vs ANZA",
        "",
        f"Status: {report['status']} ({report['patient_completed']}/5)",
        "",
        "| Patient | Status | K10 ΔclDice | CNN ΔclDice | ANZA ΔclDice | ANZA–CNN |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"]=="complete":
            def f(k): return f"{row[k]:+.5f}"
            lines.append(
                f"| {row['patient_id']} | complete | {f('K10_cldice_delta')} | "
                f"{f('conv3d_cldice_delta')} | {f('anza3d_cldice_delta')} | "
                f"{f('ANZA_minus_CNN_cldice_delta')} |"
            )
        else:
            lines.append(f"| {row['patient_id']} | {row['status']} | — | — | — | — |")
    lines+=["","## Paired patient-level comparisons",""]
    for metric,stat in report["paired_ANZA_minus_CNN"].items():
        if stat is None:
            lines.append(f"- {metric}: NOT MEASURED")
        else:
            l,h=stat["descriptive_patient_bootstrap_95pct"]
            lines.append(f"- {metric}: mean {stat['mean_paired_patient_delta']:+.5f}; descriptive bootstrap [{l:+.5f}, {h:+.5f}]")
    lines+=["","## Limitations",""]+[f"- {v}" for v in report["limitations"]]
    table.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--pilot-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--study-root",type=Path,default=DEFAULT_STUDY)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_COMPARE)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--bootstrap",type=int,default=10000)
    args=parser.parse_args()
    verify_validation_freeze(args.freeze)
    if args.run:
        for patient in sorted(VAL_CT28_IDS):
            for variant in ("conv3d","anza3d"):
                path=args.output_root/f"patient_{patient}"/variant
                if path.exists():
                    print(f"Preserving existing {patient}/{variant} evidence",flush=True)
                    continue
                try:
                    run_neural_val_patient(
                        patient=patient,variant=variant,
                        freeze_path=args.freeze,
                        pilot_root=args.pilot_root,
                        study_root=args.study_root,
                        output_root=args.output_root,
                        device=args.device,
                    )
                    print(f"Completed development-val {patient}/{variant}",flush=True)
                except Exception as exc:
                    print(f"FAILED dev-val {patient}/{variant}: {exc}",flush=True)
    report,rows=summarize_matched_validation(
        freeze_path=args.freeze,
        pilot_root=args.pilot_root,output_root=args.output_root,
        bootstrap=args.bootstrap,
    )
    write_matched(args.output_root,report,rows)
    print(json.dumps({
        "status":report["status"],
        "completed":report["patient_completed"],
        "paired":report["paired_ANZA_minus_CNN"],
    },ensure_ascii=False,indent=2))
    if args.run and report["status"]!="complete_five_patient_matched_CCTA_dev_val":
        raise RuntimeError("Neural/CT28 val comparison incomplete: inspect failed or missing patients")


if __name__=="__main__":
    main()
