"""Fixed V50/V52 paired synthetic decoder comparison; no real geology claim.

Uses exactly one model fitted on seeds 0..99 for both routes.  Test seeds
240..279 differ from the already-used V50 sanity cohort (200..239) but
come from the same renderer family.  The synthetic generating process and
artificially generated candidate endpoint proposals are not geological tests.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.tver_v50_synthetic_sanity import scene, hidden_success
from scripts.tver_v50_conditional_path import (
    fit_path_head, decode_candidate, path_heatmap,
)
from scripts.tver_v52_joint_path import infer_joint_path

TRAIN_SEEDS = tuple(range(100))
TEST_SEEDS = tuple(range(240, 280))


def run(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, receipt = fit_path_head(
        [scene(i) for i in TRAIN_SEEDS], out / "train")
    rows = []
    for seed in TEST_SEEDS:
        sample = scene(seed)
        proposal = sample["proposals"][0]
        baseline, _ = decode_candidate(
            model, sample, proposal, mode="learned_ribbon")
        heat = path_heatmap(model, sample, proposal)
        inferred = infer_joint_path(
            heat, sample["visible_prob"], sample["rectangle"],
            proposal["source"], proposal["target"])
        path = inferred.vertices if inferred is not None else None
        rows.append({
            "seed": seed,
            "v50_correct_path": int(hidden_success(sample, baseline)),
            "v52_correct_path": int(path is not None and hidden_success(sample, path)),
            "v52_abstained": int(path is None),
            "v52_hypotheses": 0 if inferred is None else inferred.candidates_considered,
        })
    totals = {
        "v50": sum(r["v50_correct_path"] for r in rows),
        "v52": sum(r["v52_correct_path"] for r in rows),
        "v52_abstained": sum(r["v52_abstained"] for r in rows),
        "v52_only_wins": sum(r["v52_correct_path"] and not r["v50_correct_path"] for r in rows),
        "v50_only_wins": sum(r["v50_correct_path"] and not r["v52_correct_path"] for r in rows),
    }
    report = {
        "status": "SYNTHETIC_SINGLE_RENDERER_ONLY_NOT_GEOLOGICAL_VALIDATION",
        "n_train": len(TRAIN_SEEDS), "n_test": len(TEST_SEEDS),
        "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        "unchanged_model_weights_between_decoders": True,
        "source_anchors_generated_from_one_synthetic_truth_family": True,
        "no_independent_geology": True,
        "no_Thebe_TEST7_access": True,
        "model_training_receipt": receipt,
        "totals": totals,
    }
    (out / "V52_SYNTHETIC_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "V52_SYNTHETIC_CASES.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "model_training_receipt"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out_dir)