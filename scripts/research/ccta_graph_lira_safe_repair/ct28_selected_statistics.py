#!/usr/bin/env python3
"""Recompute K10/15 validation claims from the selected predictions only.

Read-only, never opens CT or held-out subjects; all intervals are exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


TRAINING_TEST_FIREWALL = {954, 958, 972, 973, 980, 984}
SELECTED_VALIDATION = {957, 961, 965, 966, 974}
MODELS = ("geometry_only", "ct_conditioned")
FIELDS = {
    "model", "scene_id", "scan_id", "relation_class", "accepted", "exact",
    "false_scene", "pre_consistency_exact", "pre_consistency_false",
    "selected_relation_before_consistency", "tau_relation",
}


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(set(reader.fieldnames or [])):
            raise ValueError("Missing required selected-validation columns")
        return list(reader)


def summarize(rows: list[dict[str, str]], *, draws: int = 20000, seed: int = 20261102,
              strict_patients: bool = True) -> dict:
    if draws < 100 or not rows:
        raise ValueError("Need predictions and at least 100 bootstrap draws")
    keyed = {}
    patients = set()
    for row in rows:
        if row["model"] not in MODELS or row["relation_class"] not in {"NONE", "PAIR", "JUNCTION", "BOTH"}:
            raise ValueError("Unexpected model or relation label")
        for col in ("accepted", "exact", "false_scene", "pre_consistency_exact", "pre_consistency_false"):
            if row[col] not in {"0", "1"}:
                raise ValueError(f"Invalid boolean field {col}")
        patient = int(row["scan_id"])
        if patient in TRAINING_TEST_FIREWALL:
            raise ValueError("Held-out subject entered validation statistics")
        patients.add(patient)
        key = (row["model"], row["scene_id"])
        if key in keyed:
            raise ValueError(f"Duplicate model/scene {key}")
        keyed[key] = row
    if strict_patients and patients != SELECTED_VALIDATION:
        raise ValueError("Unexpected selected CT28 validation patient set")
    groups = {}
    for model in MODELS:
        subset = {scene: row for (m, scene), row in keyed.items() if m == model}
        groups[model] = subset
    if groups[MODELS[0]].keys() != groups[MODELS[1]].keys():
        raise ValueError("Models must evaluate exactly the same scene IDs")
    for scene in groups[MODELS[0]]:
        a, b = (groups[m][scene] for m in MODELS)
        if a["scan_id"] != b["scan_id"] or a["relation_class"] != b["relation_class"]:
            raise ValueError(f"Scene mismatch: {scene}")
    if sum(row["relation_class"] != "NONE" for row in groups[MODELS[0]].values()) == 0:
        raise ValueError("No repair-needed scenes")

    patient_ids = sorted(patients)
    def counts(model: str, selected: list[int] | None = None) -> dict:
        sample = list(groups[model].values())
        if selected is not None:
            by_patient = {}
            for row in sample:
                by_patient.setdefault(int(row["scan_id"]), []).append(row)
            sample = [row for patient in selected for row in by_patient[patient]]
        repair = [x for x in sample if x["relation_class"] != "NONE"]
        chosen = [x for x in sample if x["accepted"] == "1"]
        return {
            "scenes": len(sample),
            "repair_needed": len(repair),
            "pre_exact": sum(x["pre_consistency_exact"] == "1" for x in repair),
            "pre_false": sum(x["pre_consistency_false"] == "1" for x in sample),
            "accepted": len(chosen),
            "accepted_exact": sum(x["exact"] == "1" for x in chosen),
            "accepted_false": sum(x["false_scene"] == "1" for x in chosen),
            "by_class": {
                kind: {
                    "scenes": sum(x["relation_class"] == kind for x in sample),
                    "accepted_exact": sum(x["relation_class"] == kind and x["accepted"] == "1" and x["exact"] == "1" for x in sample),
                }
                for kind in ("PAIR", "JUNCTION", "BOTH", "NONE")
            },
        }

    totals = {model: counts(model) for model in MODELS}
    per_patient = {str(p): {model: counts(model, [p]) for model in MODELS} for p in patient_ids}
    for model, c in totals.items():
        if c["accepted"] != c["accepted_exact"] + c["accepted_false"]:
            raise ValueError(f"Accepted decisions not exact or false: {model}")

    def value(c: dict, metric: str) -> float:
        if metric == "repair_recall":
            return c["accepted_exact"] / c["repair_needed"]
        if metric == "coverage":
            return c["accepted"] / c["scenes"]
        if metric == "pre_exact_rate":
            return c["pre_exact"] / c["repair_needed"]
        raise KeyError(metric)

    rng = random.Random(seed)
    metrics = ("repair_recall", "coverage", "pre_exact_rate")
    differences = {k: [] for k in metrics}
    relevant = ("scenes", "repair_needed", "accepted", "accepted_exact", "pre_exact")
    for _ in range(draws):
        sampled = [rng.choice(patient_ids) for _ in patient_ids]
        aggregates = {
            model: {
                key: sum(per_patient[str(patient)][model][key] for patient in sampled)
                for key in relevant
            }
            for model in MODELS
        }
        for metric in metrics:
            differences[metric].append(
                value(aggregates["ct_conditioned"], metric) -
                value(aggregates["geometry_only"], metric)
            )

    def percentile(samples: list[float], quantile: float) -> float:
        ordered = sorted(samples)
        position = (len(ordered) - 1) * quantile
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (position - low) * (ordered[high] - ordered[low])

    deltas = {
        key: {
            "estimate": value(totals["ct_conditioned"], key) - value(totals["geometry_only"], key),
            "patient_cluster_bootstrap_95pct_exploratory": [percentile(values, .025), percentile(values, .975)],
        }
        for key, values in differences.items()
    }
    return {
        "status": "exploratory_selected_validation_not_held_out",
        "patient_ids": patient_ids,
        "n_independent_patients": len(patient_ids),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "models": totals,
        "per_patient": per_patient,
        "ct_minus_geometry": deltas,
        "warnings": [
            "The same five validation subjects were repeatedly used for choosing K10/15; confidence intervals do not adjust for selection bias.",
            "Scenes from one patient and perturbation repeats are not independent study subjects.",
            "Zero observed false accepted repairs is not a clinical risk guarantee.",
            "This file measures structural scene repair, not corrected 3D mask quality.",
        ],
    }


def as_markdown(report: dict) -> str:
    a = report["models"]["geometry_only"]
    b = report["models"]["ct_conditioned"]
    d = report["ct_minus_geometry"]["repair_recall"]
    lo, hi = d["patient_cluster_bootstrap_95pct_exploratory"]
    lines = [
        "# CT28 K10/15 selected-validation analysis",
        "", "| Quantity | Geometry-only | CT-conditioned |",
        "|---|---:|---:|",
        f"| Pre-gate exact / repair-needed | {a['pre_exact']}/{a['repair_needed']} | {b['pre_exact']}/{b['repair_needed']} |",
        f"| Pre-gate false / scenes | {a['pre_false']}/{a['scenes']} | {b['pre_false']}/{b['scenes']} |",
        f"| Accepted exact / repair-needed | {a['accepted_exact']}/{a['repair_needed']} | {b['accepted_exact']}/{b['repair_needed']} |",
        f"| Accepted false / all scenes | {a['accepted_false']}/{a['scenes']} | {b['accepted_false']}/{b['scenes']} |",
        f"| Coverage / scenes | {a['accepted']}/{a['scenes']} | {b['accepted']}/{b['scenes']} |",
        "", f"Paired repair-recall difference: **{100*d['estimate']:.2f} pp**; patient-cluster bootstrap 95% interval **{100*lo:.2f} to {100*hi:.2f} pp** (exploratory; n={report['n_independent_patients']} subjects).",
        "", "## Per-patient accepted exact repairs", "",
        "| Subject | Geometry-only | CT-conditioned |", "|---|---:|---:|",
    ]
    for patient, data in report["per_patient"].items():
        lines.append(f"| {patient} | {data['geometry_only']['accepted_exact']} | {data['ct_conditioned']['accepted_exact']} |")
    lines += ["", "## Interpretation limits", ""] + [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path(
        "artifacts/varvara/ct28_graph_lira/selected_validation/graph_lira_predictions_val.csv"))
    parser.add_argument("--out-json", type=Path, help="Optional output JSON")
    parser.add_argument("--out-md", type=Path, help="Optional output Markdown")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    report = summarize(read_predictions(args.predictions), draws=args.bootstrap, seed=args.seed)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(serialized + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(as_markdown(report), encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(as_markdown(report))


if __name__ == "__main__":
    main()
