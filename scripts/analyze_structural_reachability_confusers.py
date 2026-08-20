#!/usr/bin/env python3
"""Compute the predeclared secondary model-confuser audit without changing Gate A."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural_reachability.metrics import evaluate_low_fpr_curve
from structural_reachability.phase_a import FPR_MAX, OUTPUT_ROOT, RELATIONS, SEEDS


def build_confuser_audit(root: Path = OUTPUT_ROOT) -> dict[str, object]:
    metrics = json.loads((root / "metrics.json").read_text())
    if metrics.get("expert_data_accessed") is not False or metrics.get("training_performed") is not False:
        raise PermissionError("Phase A is not expert-locked and zero-training")
    rows = list(csv.DictReader((root / "per_candidate.csv").open()))
    confuser_ids = {
        int(row["pair_id"]) for row in rows
        if row["model_generated_confuser"] == "True" and int(row["label"]) == 0
    }
    if len(confuser_ids) != 20:
        raise ValueError(f"frozen confuser count drift: {len(confuser_ids)}")
    result: dict[str, object] = {
        "status": "SECONDARY_PREDECLARED_CONFUSER_AUDIT_COMPLETE",
        "primary_gate_unchanged": True,
        "primary_status": metrics["status"],
        "pair_count": len(confuser_ids),
        "selection": "top-20 validation negative scores from prior frozen classifier",
        "expert_data_accessed": False,
        "training_performed": False,
        "relations": {},
    }
    relation_output: dict[str, object] = {}
    for relation in RELATIONS:
        per_seed = {}
        for seed in SEEDS:
            selected = [
                row for row in rows
                if row["relation"] == relation
                and int(row["seed"]) == seed
                and int(row["pair_id"]) in confuser_ids
            ]
            evaluated = evaluate_low_fpr_curve(
                np.asarray([int(row["label"]) for row in selected]),
                np.asarray([float(row["score"]) for row in selected]),
                pair_ids=np.asarray([row["pair_id"] for row in selected]),
                fpr_max=FPR_MAX,
            )
            per_seed[str(seed)] = {key: value for key, value in evaluated.items() if key != "curve"}
        relation_output[relation] = {
            "per_seed": per_seed,
            "seed_mean": {
                key: float(np.mean([per_seed[str(seed)][key] for seed in SEEDS]))
                for key in (
                    "tpr_at_fpr_max", "low_fpr_partial_auc_normalized",
                    "auroc_secondary", "matched_ranking_probability",
                )
            },
        }
    result["relations"] = relation_output
    (root / "confuser_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    baseline = relation_output[RELATIONS[0]]["seed_mean"]
    full = relation_output[RELATIONS[-1]]["seed_mean"]
    (root / "SECONDARY_CONFUSER_AUDIT.md").write_text(
        "# Secondary predeclared model-confuser audit\n\n"
        "This fixed top-20 subset is descriptive only and cannot alter Gate A.\n\n"
        f"- A0 TPR@FPR<=0.05: {baseline['tpr_at_fpr_max']:.6f}\n"
        f"- A4 TPR@FPR<=0.05: {full['tpr_at_fpr_max']:.6f}\n"
        f"- A0 matched ranking: {baseline['matched_ranking_probability']:.6f}\n"
        f"- A4 matched ranking: {full['matched_ranking_probability']:.6f}\n"
        f"- Primary status remains: `{metrics['status']}`\n"
        "- Expert accessed: no; training performed: no.\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build_confuser_audit(), indent=2, sort_keys=True))
