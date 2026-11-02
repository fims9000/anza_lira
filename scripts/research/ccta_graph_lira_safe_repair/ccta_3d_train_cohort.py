#!/usr/bin/env python3
"""Run and summarize a fully reproducible train-only CCTA 3D cohort.

No validation or held-out CT28 patients can enter this pipeline.
The aggregation is read-only with respect to patient volumes and computes
paired per-patient Dice/clDice changes on existing full-mask pilot outputs.
Partial cohort failures are reported explicitly, never dropped silently.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from ccta_3d_train_gap_simulator import TRAIN_CT28_IDS
from ccta_3d_train_pilot import (
    DEFAULT_ROOT, REFERENCE_DIR, ARCHIVE, SELECTED_PACK,
    execute_train_pilot,
)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def summarize_cohort(
    *,
    root: Path, patients: list[int], bootstrap: int = 10000,
    seed: int = 20261102,
) -> tuple[dict, list[dict]]:
    if not patients or len(set(patients)) != len(patients):
        raise ValueError("Require distinct train patient IDs")
    if not set(patients) <= TRAIN_CT28_IDS:
        raise ValueError("CT28 train-only firewall: validation/test forbidden")
    if bootstrap < 100:
        raise ValueError("At least 100 resampling draws required")
    rows = []
    model_hashes = set()
    configuration = set()
    reasons = Counter()
    for patient in patients:
        patient_dir = root / f"patient_{patient}"
        manifest = patient_dir / "pilot_manifest.json"
        failure = patient_dir / "pilot_failure.json"
        record = {"patient_id": patient}
        if failure.exists():
            reason = _read(failure)
            record.update(
                status="failed",
                error_type=reason.get("error_type", "unknown"),
                error=str(reason.get("error",""))[:300],
            )
            reasons["pilot_failure"] += 1
        elif not manifest.exists():
            record.update(status="missing")
            reasons["pilot_missing"] += 1
        else:
            meta = _read(manifest)
            if meta.get("status") != "completed_experimental_train_pilot" or meta.get("patient_id") != patient:
                raise ValueError(f"Invalid patient manifest in {patient_dir}")
            if meta.get("source_kind") != "controlled_gap_from_reference_not_natural_segmentation_failure":
                raise ValueError("Mixed natural and synthetic cohort results")
            if meta.get("ground_truth_used_in_inference") is not False:
                raise ValueError("Reference segmentation leaked into solver evidence")
            if meta.get("mode") != "k10":
                raise ValueError("Cohort statistics require full K10 inference")
            decisions_path = patient_dir / "decisions.json"
            metrics_path = patient_dir / "posthoc_metrics.json"
            if not decisions_path.exists() or not metrics_path.exists():
                raise ValueError(f"Missing decision/metric file for completed {patient}")
            dec, metrics = _read(decisions_path), _read(metrics_path)
            if dec.get("patient_label") != str(patient) or metrics.get("patient_id") != str(patient):
                raise ValueError("Patient ID mismatch in post-hoc evidence")
            if metrics.get("truth_used_for_inference") is not False:
                raise ValueError("Ground truth was allegedly supplied during inference")
            digest = str(meta["model_pack_sha256"])
            model_hashes.add(digest)
            generation = meta.get("gap_generation", {})
            config = (
                generation.get("gaps"), generation.get("gap_radius_mm"),
                generation.get("seed"), dec.get("radius_mm"),
                dec.get("max_new_voxels_per_decision"),
                dec.get("path_slack_fraction"),
            )
            configuration.add(config)
            deltas = (metrics.get("dice_delta"), metrics.get("cldice_delta"))
            if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in deltas):
                raise ValueError("Missing/nonfinite Dice or clDice delta")
            for key, value in dec.get("abstention_reasons",{}).items():
                reasons["abstain:"+key] += int(value)
            if metrics.get("added_voxels") != dec.get("added_voxels"):
                raise ValueError("Added-voxel count differs between inference and metrics")
            record.update(
                status="complete", model_sha256=digest,
                scene_count=dec["scene_count"],
                endpoint_count=dec["endpoint_count"],
                accepted_decisions=dec["accepted_decisions"],
                added_voxels=dec["added_voxels"],
                false_added_voxels=metrics["reference_unsupported_added_voxels"],
                dice_before=metrics["dice_before"],
                dice_after=metrics["dice_after"],
                dice_delta=deltas[0],
                cldice_before=metrics["cldice_before"],
                cldice_after=metrics["cldice_after"],
                cldice_delta=deltas[1],
            )
        rows.append(record)
    if len(model_hashes) > 1:
        raise ValueError("Cannot aggregate distinct K10/15 model packs")
    if len(configuration) > 1:
        raise ValueError("Cannot aggregate varying gap and repair protocols as one cohort")
    complete = [row for row in rows if row["status"] == "complete"]
    draws = []
    rng = random.Random(seed)
    for metric in ("dice_delta","cldice_delta"):
        if not complete:
            draws.append((metric,None))
            continue
        values=[row[metric] for row in complete]
        resampled=sorted(sum(rng.choice(values) for _ in values)/len(values)
                         for _ in range(bootstrap))
        def q(level):
            pos=(len(resampled)-1)*level
            low=int(pos); high=min(low+1,len(resampled)-1)
            return resampled[low]+(pos-low)*(resampled[high]-resampled[low])
        draws.append((metric,{
            "mean": sum(values)/len(values),
            "exploratory_patient_bootstrap_95pct":[q(.025),q(.975)],
        }))
    report={
        "status": "complete_train_cohort" if len(complete)==len(patients) else "partial_train_cohort",
        "claim_scope": "train-only controlled synthetic gap experiments; not held-out or natural segmentation",
        "patient_count_requested":len(patients),
        "patient_count_completed":len(complete),
        "patient_count_missing":sum(row["status"]=="missing" for row in rows),
        "patient_count_failed":sum(row["status"]=="failed" for row in rows),
        "patients":patients,
        "model_sha256":next(iter(model_hashes)) if model_hashes else None,
        "fixed_protocol":list(next(iter(configuration))) if configuration else None,
        "paired_metrics_patient_macro":dict(draws),
        "total_accepted_repairs":sum(row["accepted_decisions"] for row in complete),
        "total_false_added_voxels":sum(row["false_added_voxels"] for row in complete),
        "total_added_voxels":sum(row["added_voxels"] for row in complete),
        "reasons":dict(sorted(reasons.items())),
        "bootstrap_draws":bootstrap,
        "bootstrap_seed":seed,
        "limitations":[
            "Patients without outputs are counted as missing, never imputed.",
            "Bootstrap is descriptive on a train cohort; no independent generalization claim.",
            "Voxel count is not a patient-level false-bridge rate.",
            "CCTA generator may be out of distribution relative to K10/15 controlled five-endpoint training.",
        ],
    }
    return report,rows


def write_report(root: Path, report: dict, rows: list[dict]) -> None:
    destinations=(root/"train_cohort_summary.json",
                  root/"train_cohort_patients.csv",
                  root/"train_cohort_results.md")
    if any(p.exists() for p in destinations):
        raise FileExistsError("Cohort report already exists; immutable snapshots only")
    root.mkdir(parents=True,exist_ok=True)
    destinations[0].write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                               encoding="utf-8")
    columns=sorted({key for row in rows for key in row})
    with destinations[1].open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines=[
        "# CT28 full-volume train-only CCTA repair",
        "",
        f"Status: **{report['status']}**. Complete: {report['patient_count_completed']}/{report['patient_count_requested']}.",
        "",
        "| Patient | Status | Scenes | Repairs | Added voxels | Dice before | Dice after | Delta clDice |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in rows:
        def f(k):
            return f"{x[k]:.4f}" if isinstance(x.get(k),float) else str(x.get(k,"—"))
        lines.append(f"| {x['patient_id']} | {x['status']} | {f('scene_count')} | {f('accepted_decisions')} | {f('added_voxels')} | {f('dice_before')} | {f('dice_after')} | {f('cldice_delta')} |")
    lines += [
        "", "## Paired patient-macro improvements (train only)",
        ""
    ]
    for metric,result in report["paired_metrics_patient_macro"].items():
        if result:
            low,high=result["exploratory_patient_bootstrap_95pct"]
            lines.append(f"- {metric}: mean {result['mean']:+.5f}, descriptive bootstrap 95% [{low:+.5f}, {high:+.5f}]")
    lines += [
        "", f"Added voxels: {report['total_added_voxels']}; outside GT: {report['total_false_added_voxels']}.",
        "", "## Limitations", "",
    ] + [f"- {x}" for x in report["limitations"]]
    destinations[2].write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT/"train_cohort")
    parser.add_argument("--patients",type=int,nargs="+",default=sorted(TRAIN_CT28_IDS))
    parser.add_argument("--run",action="store_true",
                        help="Execute each missing train-patient pilot using local CT and K10 pack")
    parser.add_argument("--bootstrap",type=int,default=10000)
    parser.add_argument("--gaps",type=int,default=3)
    parser.add_argument("--gap-radius-mm",type=float,default=1.5)
    parser.add_argument("--repair-radius-mm",type=float,default=0.)
    parser.add_argument("--max-new-voxels",type=int,default=250)
    parser.add_argument("--model-pack",type=Path,default=SELECTED_PACK)
    parser.add_argument("--archive-dir",type=Path,default=ARCHIVE)
    args=parser.parse_args()
    if len(set(args.patients))!=len(args.patients) or not set(args.patients)<=TRAIN_CT28_IDS:
        raise ValueError("Only distinct CT28 train patients allowed")
    if args.run:
        args.root.mkdir(parents=True,exist_ok=True)
        for patient in args.patients:
            target=args.root/f"patient_{patient}"
            if target.exists():
                print(f"Preserving existing patient pilot {patient}",flush=True)
                continue
            try:
                execute_train_pilot(
                    scan_id=patient, output_dir=target,
                    archive_dir=args.archive_dir,model_pack=args.model_pack,
                    mode="k10",gaps=args.gaps,gap_radius_mm=args.gap_radius_mm,
                    repair_radius_mm=args.repair_radius_mm,
                    max_new_voxels=args.max_new_voxels,
                )
                print(f"Completed patient {patient}",flush=True)
            except Exception as error:
                print(f"FAILED patient {patient}: {error}",flush=True)
    report,rows=summarize_cohort(
        root=args.root,patients=args.patients,bootstrap=args.bootstrap
    )
    write_report(args.root,report,rows)
    print(json.dumps({k:v for k,v in report.items() if k in (
        "status","patient_count_requested","patient_count_completed",
        "patient_count_failed","patient_count_missing",
        "paired_metrics_patient_macro")},indent=2))
    if args.run and report["patient_count_completed"] != report["patient_count_requested"]:
        raise RuntimeError("Train cohort is incomplete: inspect missing/failed pilot records")


if __name__=="__main__":
    main()
