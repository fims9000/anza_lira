#!/usr/bin/env python3
"""Read-only diagnostics for the frozen CT28 controlled scene protocol.

This reads only train/validation scene metadata; it never opens CT or held-out test.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_scene_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]
    if len(rows) != len({row["scene_id"] for row in rows}):
        raise ValueError("Duplicate scene IDs")
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Unexpected split")

    lookup = defaultdict(Counter)
    for row in train:
        lookup[(row["pair_candidate_count"], row["junction_candidate_count"])][
            row["relation_class"]
        ] += 1

    correct = 0
    covered = 0
    by_class = defaultdict(lambda: {"n": 0, "covered": 0, "correct": 0})
    for row in validation:
        category = row["relation_class"]
        by_class[category]["n"] += 1
        counts = lookup.get((row["pair_candidate_count"], row["junction_candidate_count"]))
        if counts is None:
            continue
        covered += 1
        by_class[category]["covered"] += 1
        guess = max(sorted(counts), key=lambda label: counts[label])
        matched = int(guess == category)
        correct += matched
        by_class[category]["correct"] += matched

    pattern_counts = defaultdict(Counter)
    for row in rows:
        k = (row["split"], row["scene_family"])
        pair = row["true_pair_endpoint_ids"] or "NONE"
        junction = row["true_junction_endpoint_ids"] or "NONE"
        pattern_counts[k][(pair, junction)] += 1

    families = {
        f"{split}/{family}": {
            "n": sum(counter.values()),
            "patterns": [
                {"pair": pair, "junction": junction, "n": n}
                for (pair, junction), n in sorted(counter.items())
            ],
        }
        for (split, family), counter in sorted(pattern_counts.items())
    }
    return {
        "scope": "controlled train/validation scene metadata, not independent test",
        "train_scenes": len(train),
        "validation_scenes": len(validation),
        "validation_majority_class_accuracy": (
            max(Counter(row["relation_class"] for row in validation).values()) /
            len(validation)
        ),
        "count_only_train_lookup": {
            "covered": covered,
            "total": len(validation),
            "correct": correct,
            "accuracy_on_covered": correct / covered,
            "by_class": dict(by_class),
        },
        "endpoint_identity_patterns": families,
        "limitations": [
            "This is a shortcut diagnostic, not evidence that the trained head uses endpoint IDs.",
            "No counterfactual scene regeneration was performed.",
            "Candidate pools are frozen during saved perturbation evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-plan",
        type=Path,
        default=Path("artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv"),
    )
    args = parser.parse_args()
    print(json.dumps(audit_scene_protocol(args.scene_plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
