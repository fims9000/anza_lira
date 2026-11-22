"""V54: frozen-policy versus expanded SBPP candidate coverage on Thebe TRAIN/VAL.

Exploratory, label-blind proposal generation. The already-inspected REL_CAL is
diagnostic only. Does not open TEST7, retrain models, or select a deployment policy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.tver_v26_predicted_cracks_e2e import (
    is_true_relation, sbpp_proposals,
)
from scripts.tver_v43_dense_mask_fallback import (
    DENSE_STOP, FROZEN_ANZA_SHA, adaptive_threshold,
)

# Freeze these engineering diagnostics before the V54 evaluation.
POLICIES = {
    "V53_frozen_4x12": dict(max_sources=4, per_source_k=12,
                            max_pairs=48, max_source_radius=29.,
                            min_alignment=.10),
    "V54_k_only_4x24": dict(max_sources=4, per_source_k=24,
                            max_pairs=96, max_source_radius=29.,
                            min_alignment=.10),
    "V54_sources_only_8x12": dict(max_sources=8, per_source_k=12,
                                  max_pairs=96, max_source_radius=29.,
                                  min_alignment=.10),
    "V54_sources_and_k_8x24": dict(max_sources=8, per_source_k=24,
                                   max_pairs=192, max_source_radius=29.,
                                   min_alignment=.10),
    "V54_relaxed_8x24": dict(max_sources=8, per_source_k=24,
                             max_pairs=192, max_source_radius=40.,
                             min_alignment=-.10),
}

EXPECTED = {"TRAIN": (100, 200, 125), "REL_CAL_ALREADY_VIEWED": (60, 120, 75)}


def _policy_scene(scene, name, policy):
    # No expert information reaches any proposal method. Truth is consulted only
    # after proposals are built and immutable for this sample.
    threshold = (
        adaptive_threshold(scene["visible_prob"], .70)
        if scene["status"] == DENSE_STOP else .65
    )
    prediction = sbpp_proposals(
        scene["visible_prob"], scene["gray"], scene["rectangle"], threshold,
        diagnostics=True, **policy,
    )
    proposals = prediction["proposals"]
    if any("y" in p or "true" in p or "expert" in p for p in proposals):
        raise RuntimeError("Expert label entered proposal generation")
    hits = (
        [is_true_relation(p, scene["true_ends"], scene["origin"])
         for p in proposals]
        if scene["kind"] == "LINK" else []
    )
    diag = prediction.get("diagnostics", {})
    return {
        "section": int(scene["section"]), "gap_id": scene["id"],
        "world": scene["kind"], "policy": name, "status": prediction["status"],
        "adaptive_tau": float(threshold),
        "n_candidates": len(proposals),
        "n_proposals_before_cap": int(prediction.get("n_proposals_before_cap", 0)),
        "candidate_available": int(any(hits)),
        "geometric_top1_correct": int(bool(hits) and bool(hits[0])),
        "ports_total": int(diag.get("ports_total", 0)),
        "ports_after_radius_filter": int(diag.get("ports_after_radius_filter", 0)),
        "ports_after_direction_filter": int(diag.get("ports_after_direction_filter", 0)),
        "ports_used": int(diag.get("ports_used", 0)),
        "proposals_before_per_source_k": int(diag.get("proposals_before_per_source_k", 0)),
        "proposals_after_per_source_k": int(diag.get("proposals_after_per_source_k", 0)),
        "final_cap": int(policy["max_pairs"]),
    }


def evaluate_scenes(scenes, cohort, frozen_rows=None):
    expected_sections, expected_link, expected_available = EXPECTED[cohort]
    pos = [s for s in scenes if s["kind"] == "LINK"]
    neg = [s for s in scenes if s["kind"] == "NO_LINK"]
    ids = lambda a: {(int(s["section"]), str(s["id"])) for s in a}
    if (len(pos) != expected_link or len(neg) != expected_link
            or ids(pos) != ids(neg)
            or len({s["section"] for s in pos}) != expected_sections):
        raise RuntimeError("Source cohort or controlled world drift")
    rows = [
        _policy_scene(scene, name, policy)
        for scene in scenes for name, policy in POLICIES.items()
    ]
    grouped = {
        name: [r for r in rows if r["policy"] == name and r["world"] == "LINK"]
        for name in POLICIES
    }
    legacy = grouped["V53_frozen_4x12"]
    if sum(r["candidate_available"] for r in legacy) != expected_available:
        raise RuntimeError("Original V53 candidate recall not reproduced")
    if frozen_rows is not None:
        exact = {
            (int(r["section"]), str(r["gap_id"])): r
            for r in frozen_rows
            if r["method"] == "raw_pair_rank_score" and r["world"] == "LINK"
        }
        if len(exact) != expected_link:
            raise RuntimeError("V53 artifact source cases mismatch")
        for row in legacy:
            match = exact[(row["section"], row["gap_id"])]
            if (row["n_candidates"] != int(match["n_candidates"])
                    or row["candidate_available"] != int(match["true_candidate_available"])):
                raise RuntimeError("Original V53 candidate receipt drift")
    baseline = {(r["section"], r["gap_id"]): r for r in legacy}
    summaries = {}
    for name in POLICIES:
        link = grouped[name]
        no_link = [r for r in rows if r["policy"] == name and r["world"] == "NO_LINK"]
        rescue = [r for r in link if
                  r["candidate_available"] and
                  not baseline[(r["section"], r["gap_id"])]["candidate_available"]]
        lost = [r for r in link if
                not r["candidate_available"] and
                baseline[(r["section"], r["gap_id"])]["candidate_available"]]
        summaries[name] = {
            "n_link_cases": len(link),
            "n_source_sections": expected_sections,
            "correct_candidate_available": sum(r["candidate_available"] for r in link),
            "candidate_recall": sum(r["candidate_available"] for r in link) / len(link),
            "rescued_legacy_misses": len(rescue),
            "lost_legacy_hits": len(lost),
            "rescued_ids": [r["gap_id"] for r in rescue],
            "lost_ids": [r["gap_id"] for r in lost],
            "geometric_top1_correct": sum(r["geometric_top1_correct"] for r in link),
            "median_candidates": float(np.median([r["n_candidates"] for r in link])),
            "p95_candidates": float(np.percentile([r["n_candidates"] for r in link], 95)),
            "max_candidates": max(r["n_candidates"] for r in link),
            "at_final_cap": sum(r["n_candidates"] == r["final_cap"] for r in link),
            "before_final_cap_gt_final_cap": sum(
                r["n_proposals_before_cap"] > r["final_cap"] for r in link),
            "no_link_cases_with_candidates": sum(r["n_candidates"] > 0 for r in no_link),
            "mean_no_link_candidates": float(np.mean(
                [r["n_candidates"] for r in no_link])),
            "candidate_count_only_not_false_link_rate": True,
        }
    return rows, summaries


def execute(memmap_root, frontends_dir, v53_receipts, out_dir):
    import torch

    from scripts.tver_v34_anza_p0_relation_cal import _state_report
    from scripts.tver_v33_thebe_train_segmenters import read_memmap
    from scripts.tver_v33_thebe_relation_cal import (
        TRAIN_SOURCE_IDS, RELATION_CAL_SOURCE_IDS, _candidate_scenes,
        _load_model,
    )

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = _state_report(frontends_dir)
    if state["frontends"]["anza_v1"]["SHA256"] != FROZEN_ANZA_SHA:
        raise RuntimeError("V33 original trained ANZA SHA mismatch")
    if not state["heldout_THEBE_TEST7_NOT_OPENED"]:
        raise RuntimeError("Original TEST7 lock missing")
    model = _load_model(frontends_dir, "anza_v1", state)
    receipt_path = Path(v53_receipts) / "V46_CASE_ROWS.csv"
    with receipt_path.open(newline="", encoding="utf-8") as f:
        frozen_rows = list(csv.DictReader(f))

    output_rows, summary = [], {}
    for cohort, raw_name, expert_name, section_ids in (
        ("TRAIN", "train_seismic_float32.npy", "train_fault_bool.npy",
         TRAIN_SOURCE_IDS),
        ("REL_CAL_ALREADY_VIEWED", "val_seismic_float32.npy", "val_fault_bool.npy",
         RELATION_CAL_SOURCE_IDS),
    ):
        seismic = read_memmap(memmap_root, raw_name)
        expert = read_memmap(memmap_root, expert_name)
        scenes, _ = _candidate_scenes(
            seismic, expert, section_ids, state["normalization"], model, .65)
        part, result = evaluate_scenes(
            scenes, cohort, frozen_rows if cohort == "REL_CAL_ALREADY_VIEWED" else None)
        output_rows += part
        summary[cohort] = result
        print("V54_CANDIDATE_COHORT", cohort, json.dumps(
            {name: {k: v for k, v in stats.items() if k not in
                     ("rescued_ids", "lost_ids")}
             for name, stats in result.items()}, sort_keys=True), flush=True)
        del scenes, seismic, expert

    report = {
        "status": "V54_EXPLORATORY_FROZEN_V53_TRACE_CANDIDATE_DIAGNOSTIC",
        "policies_predeclared": POLICIES,
        "v53_checkpoint_sha256": FROZEN_ANZA_SHA,
        "v53_candidate_receipt_verified": True,
        "one_field_already_inspected_REL_CAL_not_independent_confirm": True,
        "synthetic_no_link_not_real_geological_instance_negative": True,
        "all_proposals_label_blind": True,
        "no_TEST7_access": True,
        "generator_coverage_only_not_end_to_end_recall": True,
        "summary": summary,
    }
    (out / "V54_CANDIDATE_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    with (out / "V54_CANDIDATE_CASES.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return report


def main():
    p = argparse.ArgumentParser()
    for arg in ("mmap-root", "frontends-dir", "v53-receipts", "out-dir"):
        p.add_argument("--" + arg, type=Path, required=True)
    opts = p.parse_args()
    execute(opts.mmap_root, opts.frontends_dir, opts.v53_receipts, opts.out_dir)


if __name__ == "__main__":
    main()
