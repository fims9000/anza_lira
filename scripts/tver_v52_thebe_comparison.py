"""Paired, posthoc V50 vs V52 controlled-gap comparison on Thebe REL-CAL.

The evaluation never opens TEST7 or selects parameters on REL-CAL. The frozen
original V50 train/candidate/ranker sequence runs first; this module receives
its TRAIN-fitted models and identical CAL scene objects, with the historical
per-case V50 results. It only evaluates reconstructed paths after inference.

REL-CAL has already been inspected in previous experiments and consists of
source sections in ONE seismic volume, so cluster CIs are descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.tver_v52_joint_path import decode_candidate_v52

BOOTSTRAP_SEED = 20261116
BOOTSTRAP_REPEATS = 5000
EXPECTED_SLICES = 60
EXPECTED_LINK = 120
EXPECTED_NO_LINK = 120


def _ranked_choice(scene, scale, ranker):
    """Rank using the same frozen V50 pairwise ranker; no truth at inference."""
    from scripts.tver_v40_pairwise_ranker import pair_rank_scores
    scores = pair_rank_scores(scale, ranker, scene)
    if len(scores) != len(scene["proposals"]):
        raise RuntimeError("Ranker did not score exactly the frozen candidate list")
    if not len(scores):
        return None
    return scene["proposals"][int(np.argmax(scores))]


def _descriptive_source_cluster_ci(rows, *, seed=BOOTSTRAP_SEED,
                                   repetitions=BOOTSTRAP_REPEATS):
    """Paired section resampling, NOT independent-geological-field inference."""
    by_source = {}
    for case in rows:
        if case["world"] != "LINK":
            continue
        by_source.setdefault(case["section"], []).append(case)
    if len(by_source) != EXPECTED_SLICES or any(len(cases) != 2 for cases in by_source.values()):
        raise RuntimeError("Expected 60 source sections with two paired LINK gaps each")
    ids = sorted(by_source)
    deltas = np.asarray([
        sum(case["v52_path_correct"] - case["v50_path_correct"] for case in by_source[sid])
        for sid in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ids), size=(repetitions, len(ids)))
    # Success-count difference per 120 LINK cases, in percentage points.
    effects = deltas[sampled].sum(axis=1) / EXPECTED_LINK * 100.
    bounds = np.quantile(effects, [.025, .975])
    return {
        "unit": "source_section_with_two_LINK_gaps",
        "n_clusters": len(ids),
        "seed": seed,
        "bootstrap_repetitions": repetitions,
        "observed_success_rate_delta_percentage_points": float(deltas.sum() / EXPECTED_LINK * 100),
        "descriptive_cluster_bootstrap_95pct_ci_percentage_points": list(map(float, bounds)),
        "not_independent_field_confidence_interval": True,
    }


def _hidden_path_scorer(scene, path):
    from scripts.tver_v49_boundary_path import hidden_path_success
    return hidden_path_success(scene, path)


def evaluate_paired_cal(scenes, scale, ranker, pathnet, v50_evidence, out_dir,
                        *, scorer=_hidden_path_scorer, decoder=decode_candidate_v52,
                        select=_ranked_choice):
    """Emit per-world receipts and independent of oracle *decoder* scoring.

    `scorer` reads expert centerline only AFTER the decoder returns its route.
    Historical V50 per-case successes are reused verbatim, never recomputed.
    Test callbacks permit synthetic contract tests without Thebe downloads.
    """
    baseline_rows = v50_evidence["case_rows"]
    frozen = {(int(r["section"]), str(r["gap_id"])): r for r in baseline_rows}
    if len(frozen) != EXPECTED_LINK:
        raise RuntimeError("Wrong fixed V50 controlled LINK cohort")
    cases = []
    for scene in scenes:
        world = str(scene["kind"])
        if world not in ("LINK", "NO_LINK"):
            raise RuntimeError("Unknown synthetic intervention world")
        section, gid = int(scene["section"]), str(scene["id"])
        proposal = select(scene, scale, ranker)
        if proposal is None:
            path, meta = None, {"status": "ABSTAIN_NO_PAIRS"}
        else:
            path, meta = decoder(pathnet, scene, proposal)
        if path is not None:
            if not len(path) or not all(0 <= y < 96 and 0 <= x < 96 for y, x in path):
                raise RuntimeError("V52 returned invalid raster trajectory")
            for (y0, x0), (y1, x1) in zip(path, path[1:]):
                if max(abs(y1-y0), abs(x1-x0)) > 1:
                    raise RuntimeError("V52 returned a disconnected raster trajectory")
        pair_right = bool(world == "LINK" and proposal is not None and int(proposal["y"]) == 1)
        # Truth is read only by the metric, never by the model or decoder.
        passed = bool(pair_right and path is not None and scorer(scene, path))
        base = frozen.get((section, gid)) if world == "LINK" else None
        if world == "LINK" and base is None:
            raise RuntimeError("V52 changed original Thebe LINK case identity")
        if base is not None and bool(base["correct_top1_pair"]) != pair_right:
            raise RuntimeError("V52 top1 candidate differs from historical V50")
        cases.append({
            "section": section, "gap_id": gid, "world": world,
            "top1_correct": int(pair_right),
            "v50_path_correct": int(base["successful_learned_ribbon_gap_path"]) if base else 0,
            "v52_path_correct": int(passed),
            "v52_route_emitted": int(path is not None),
            "v52_abstained": int(path is None),
            "v52_decoder_status": str(meta["status"]),
            "v52_hypotheses": int(meta.get("n_hypotheses", 0)),
        })
    pos = [row for row in cases if row["world"] == "LINK"]
    neg = [row for row in cases if row["world"] == "NO_LINK"]
    if (len(pos), len(neg)) != (EXPECTED_LINK, EXPECTED_NO_LINK):
        raise RuntimeError("V52 full controlled cohort cardinality changed")
    if len({row["section"] for row in pos}) != EXPECTED_SLICES:
        raise RuntimeError("Incorrect number of Thebe CAL source sections")
    v50_good = sum(row["v50_path_correct"] for row in pos)
    v52_good = sum(row["v52_path_correct"] for row in pos)
    if v50_good != 18 or sum(row["top1_correct"] for row in pos) != 45:
        raise RuntimeError("Original frozen V50 real positive count drifted")
    report = {
        "status": "V52_POSTHOC_ORIGINAL_THEBE_REL_CAL_SINGLE_FIELD_TEST7_UNOPENED",
        "label_blind_decoder_and_fixed_v50_training": True,
        "already_inspected_rel_cal_not_independent_test": True,
        "known_expert_localized_synthetic_gap_intervention": True,
        "raw_seismic_still_visible_inside_gap": True,
        "no_real_world_false_link_certification": True,
        "no_TEST7_access": True,
        "n_LINK": len(pos), "n_synthetic_NO_LINK": len(neg),
        "n_source_sections": EXPECTED_SLICES,
        "n_original_correct_top1_pairs": 45,
        "path_success_definition": "precision >= .8 AND recall >= .8 within 2px of expert gap centerline",
        "n_original_v50_correct_paths": v50_good,
        "n_v52_correct_paths": v52_good,
        "n_v52_abstain_LINK": sum(row["v52_abstained"] for row in pos),
        "n_v52_abstain_NO_LINK": sum(row["v52_abstained"] for row in neg),
        "n_v52_wrong_LINK_paths_emitted": sum(row["v52_route_emitted"] and not row["v52_path_correct"] for row in pos),
        "n_v52_synthetic_NO_LINK_paths_emitted": sum(row["v52_route_emitted"] for row in neg),
        "paired_wins_v52_not_v50": sum(row["v52_path_correct"] and not row["v50_path_correct"] for row in pos),
        "paired_wins_v50_not_v52": sum(row["v50_path_correct"] and not row["v52_path_correct"] for row in pos),
        "source_cluster_bootstrap": _descriptive_source_cluster_ci(cases),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "V52_REAL_THEBE_REL_CAL_RESULT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    with (out / "V52_REAL_THEBE_REL_CAL_CASES.csv").open("w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print("V52_REAL_THEBE_REL_CAL_RESULTS", json.dumps(report, indent=2), flush=True)
    return report


def main():
    from scripts.tver_v50_path_net_experiment import execute as v50_execute
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmap-root", type=Path, required=True)
    parser.add_argument("--frontends-dir", type=Path, required=True)
    parser.add_argument("--v43-receipt-dir", type=Path, required=True)
    parser.add_argument("--v45-receipt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    def run_comparison(scenes, scaler, ranker, pathnet, v50_evidence, out, **unused):
        return evaluate_paired_cal(scenes, scaler, ranker, pathnet, v50_evidence, out)
    v50_execute(args.mmap_root, args.frontends_dir,
                args.v43_receipt_dir, args.v45_receipt_dir, args.out_dir,
                comparison_hook=run_comparison)


if __name__ == "__main__":
    main()