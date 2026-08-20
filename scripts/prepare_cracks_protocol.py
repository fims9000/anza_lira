#!/usr/bin/env python3
"""Freeze the ANZA-LIRA v2 CRACKS crowd-to-expert data contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_cracks_archives import DATA_ROOT, RESULT_ROOT, write_json


V2_RESULT_ROOT = PROJECT_ROOT / "results" / "anza_v2_study"
DEFAULT_INVENTORY = RESULT_ROOT / "archive_inventory.json"
DEFAULT_COLOR_AUDIT = RESULT_ROOT / "expert_color_audit.json"
DEFAULT_FOLDS = DATA_ROOT / "splits" / "anza_v2_folds.json"
HOLDOUT_SALT = "anza-v2-crowd-holdout-v1"
FOLD_PROTOCOL = "numeric-contiguous-test-cyclic-next-validation-v1"
MIN_ANNOTATOR_SECTIONS = 300

WHITE = (255, 255, 255)
ORANGE = (255, 127, 14)
GREEN = (44, 160, 44)
BLUE = (31, 119, 180)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_rank(names: Iterable[str], *, salt: str) -> list[str]:
    return sorted(names, key=lambda name: (hashlib.sha256(f"{salt}:{name}".encode()).hexdigest(), name))


def select_crowd_holdout(
    annotation_counts: Mapping[str, int],
    *,
    minimum_sections: int = MIN_ANNOTATOR_SECTIONS,
) -> dict[str, Any]:
    eligible_novices = [
        name for name, count in annotation_counts.items() if name.startswith("novice") and count >= minimum_sections
    ]
    eligible_practitioners = [
        name
        for name, count in annotation_counts.items()
        if name.startswith("practitioner") and count >= minimum_sections
    ]
    if len(eligible_novices) < 2 or not eligible_practitioners:
        raise ValueError("Insufficient annotator coverage for the required 2-novice/1-practitioner holdout")
    novices = stable_rank(eligible_novices, salt=f"{HOLDOUT_SALT}:novice")[:2]
    practitioners = stable_rank(eligible_practitioners, salt=f"{HOLDOUT_SALT}:practitioner")[:1]
    held_out = sorted(novices + practitioners)
    return {
        "selection": "stable SHA-256 rank of annotator ID; never model-performance selected",
        "salt": HOLDOUT_SALT,
        "minimum_sections": minimum_sections,
        "novices": novices,
        "practitioners": practitioners,
        "all": held_out,
        "coverage": {name: int(annotation_counts[name]) for name in held_out},
    }


def build_expert_folds(expert_ids: Sequence[int]) -> list[dict[str, Any]]:
    ordered = sorted({int(value) for value in expert_ids})
    if len(ordered) != 40:
        raise ValueError(f"ANZA-LIRA v2 Setting B requires exactly 40 available expert sections, got {len(ordered)}")
    folds: list[dict[str, Any]] = []
    for fold_index in range(5):
        test = ordered[8 * fold_index : 8 * (fold_index + 1)]
        remaining_cycle = ordered[8 * (fold_index + 1) :] + ordered[: 8 * fold_index]
        validation = remaining_cycle[:4]
        train = sorted(set(ordered) - set(test) - set(validation))
        if (len(train), len(validation), len(test)) != (28, 4, 8):
            raise AssertionError("Invalid expert fold cardinality")
        if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
            raise AssertionError("Expert fold overlap")
        folds.append(
            {
                "fold": fold_index,
                "train": train,
                "validation": validation,
                "test": test,
                "setting_c_excluded_section_ids": sorted(
                    {neighbor for section in test for neighbor in range(section - 2, section + 3) if 1 <= neighbor <= 400}
                ),
            }
        )
    test_counter = Counter(section for fold in folds for section in fold["test"])
    if test_counter != Counter({section: 1 for section in ordered}):
        raise AssertionError("Each available expert section must appear in exactly one Setting B test fold")
    return folds


def mask_policies() -> dict[str, Any]:
    return {
        "paper_like": {
            "positive_rgb": [list(BLUE), list(GREEN)],
            "negative_rgb": [list(WHITE)],
            "ignore_rgb": [list(ORANGE)],
            "basis": (
                "The paper combines certain and uncertain faults and excludes orange no-fault labels. "
                "White palette index 0 is treated as canvas/background only in this named sensitivity policy."
            ),
            "white_status": "INFERRED_CANVAS_BACKGROUND_NOT_EXPLICITLY_DOCUMENTED",
        },
        "conservative": {
            "positive_rgb": [list(BLUE), list(GREEN)],
            "negative_rgb": [list(ORANGE)],
            "ignore_rgb": [list(WHITE)],
            "basis": "Only the three officially documented semantic colors are assigned; undocumented white is ignored.",
            "white_status": "UNASSIGNED_IGNORE",
        },
    }


def expert_availability(inventory: Mapping[str, Any]) -> dict[str, Any]:
    annotations = inventory["annotations"]
    directories = annotations["annotation_directories"]
    expert_like = sorted(name for name in directories if "expert" in name.lower())
    expert_ids = [int(value) for value in annotations["expert_section_ids"]]
    return {
        "status": "AVAILABLE_EXPERT_SUBSET_CONFIRMED",
        "available_expert_subset_sections": len(expert_ids),
        "available_expert_section_ids": sorted(expert_ids),
        "expert_like_released_directories": expert_like,
        "other_released_expert_directory_found": expert_like != ["expert"],
        "released_expert_test_masks_found": any(name.lower() != "expert" for name in expert_like),
        "paper_consistency": {
            "paper_expert_training_sections": 40,
            "paper_expert_training_faults": 815,
            "paper_expert_test_faults": 6821,
            "count_interpretation": "CONSISTENT_WITH_PAPER_TRAIN_SUBSET_COUNT",
            "identity_interpretation": "SECTION_ID_EQUIVALENCE_NOT_PROVEN_BY_RELEASE_METADATA",
            "public_test_interpretation": "PAPER_TEST_MASKS_NOT_PRESENT_IN_RELEASED_ARCHIVE",
        },
        "evidence": {
            "paper": "https://arxiv.org/abs/2408.11185",
            "official_repository": "https://github.com/olivesgatech/CRACKS",
            "official_repository_commit_audited": "faaf7c32f1f23ba66130455101809010e91b3c9f",
            "archive_inventory": "results/cracks_study/archive_inventory.json",
        },
    }


def _semantics_markdown(color_audit: Mapping[str, Any], policies: Mapping[str, Any]) -> str:
    rows = {tuple(row["rgb"]): row for row in color_audit["colors"]}
    definitions = [
        (BLUE, "fault certain", "Official paper and repository", "positive", "positive"),
        (GREEN, "fault uncertain", "Official paper and repository", "positive", "positive"),
        (ORANGE, "no fault certain", "Official paper; excluded from its main experiments", "ignore", "negative"),
        (
            WHITE,
            "not documented; palette index 0 / untouched canvas candidate",
            "Observed archive palette; official code does not resolve it",
            "negative",
            "ignore",
        ),
    ]
    lines = [
        "# CRACKS mask semantics",
        "",
        "Status: `TWO_EXPLICIT_POLICIES_FROZEN`",
        "",
        "No undocumented RGB value is silently converted into ground truth. The official",
        "repository commit audited was `faaf7c32f1f23ba66130455101809010e91b3c9f`.",
        "Its released preprocessing code does not establish a semantic meaning for white.",
        "",
        "| RGB | Observed pixels (expert subset) | Official meaning | Evidence | paper_like | conservative |",
        "|---|---:|---|---|---|---|",
    ]
    for rgb, meaning, evidence, paper_value, conservative_value in definitions:
        lines.append(
            f"| `{rgb}` | {int(rows[rgb]['pixel_count'])} | {meaning} | {evidence} | "
            f"{paper_value} | {conservative_value} |"
        )
    lines.extend(
        [
            "",
            "## Policies",
            "",
            "- `paper_like`: certain and uncertain fault are positive, white is inferred canvas/background,",
            "  and orange is ignored, matching the paper's stated exclusion of orange labels.",
            "- `conservative`: certain and uncertain fault are positive, documented orange is negative,",
            "  and undocumented white is ignored.",
            "",
            "Both policies are evaluated as a sensitivity analysis. Neither may be renamed after results.",
            "",
            "Machine-readable definitions are stored in `results/anza_v2_study/protocol.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_protocol(inventory: Mapping[str, Any], color_audit: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    annotations = inventory["annotations"]
    section_ids_by_annotator = annotations["section_ids_by_annotator"]
    image_ids = {int(value) for value in inventory["images"]["image_identifiers"]}
    crowd_names = sorted(
        name for name in annotations["annotation_directories"] if name.startswith(("novice", "practitioner"))
    )
    paired_counts = {
        name: len(image_ids & {int(value) for value in section_ids_by_annotator.get(name, [])})
        for name in crowd_names
    }
    holdout = select_crowd_holdout(paired_counts)
    train_annotators = sorted(set(crowd_names) - set(holdout["all"]))
    if "expert" in train_annotators:
        raise AssertionError("Expert labels entered Setting A training annotators")
    train_sections = sorted(
        image_ids & {int(section) for name in train_annotators for section in section_ids_by_annotator.get(name, [])}
    )
    validation_sections = sorted(
        image_ids & {int(section) for name in holdout["all"] for section in section_ids_by_annotator.get(name, [])}
    )
    all_crowd_sections = {
        int(section) for name in crowd_names for section in section_ids_by_annotator.get(name, [])
    }
    folds = build_expert_folds(annotations["expert_section_ids"])
    fold_payload = {
        "protocol": FOLD_PROTOCOL,
        "available_expert_subset": sorted(int(value) for value in annotations["expert_section_ids"]),
        "folds": folds,
    }
    fold_payload["sha256"] = canonical_hash(fold_payload)
    policies = mask_policies()
    availability = expert_availability(inventory)
    protocol: dict[str, Any] = {
        "study": "ANZA-LIRA v2",
        "status": "FROZEN_BEFORE_TRAINING",
        "source_archive_status": "VERIFIED",
        "expert_availability": availability,
        "mask_policies": policies,
        "confidence_weights": {"certain_fault": 1.5, "uncertain_fault": 1.0},
        "expertise_weights": {"practitioner": 2.0, "novice": 1.0},
        "setting_a": {
            "name": "crowd-to-expert same-image transfer",
            "training_annotators": train_annotators,
            "held_out_annotators": holdout,
            "training_section_count": len(train_sections),
            "training_section_ids": train_sections,
            "held_out_validation_section_count": len(validation_sections),
            "held_out_validation_section_ids": validation_sections,
            "orphan_crowd_annotation_section_ids": sorted(all_crowd_sections - image_ids),
            "images_without_any_crowd_annotation": sorted(image_ids - all_crowd_sections),
            "expert_gradient": False,
            "expert_evaluation_sections": availability["available_expert_section_ids"],
            "claim_boundary": "not unseen-section generalization",
        },
        "setting_b": {
            "name": "crowd pretraining plus limited-expert fine-tuning",
            "fold_manifest_sha256": fold_payload["sha256"],
            "fold_count": 5,
            "per_fold_counts": {"train": 28, "validation": 4, "test": 8},
            "max_epochs": 20,
        },
        "setting_c": {
            "name": "image-disjoint robustness",
            "models": ["unet", "anza_v1", "anza_v2_full"],
            "seed": 42,
            "neighbor_guard": 2,
            "exclusion": "held-out images and every annotation for them, plus available +/-2 section neighbors",
        },
        "test_access": {
            "expert_scores_unlocked": False,
            "synthetic_test_unlocked": False,
            "unlock_requires": "frozen architecture, thresholds, trace parameters, and provenance record",
        },
    }
    protocol["sha256"] = canonical_hash(protocol)
    return protocol, fold_payload


def prepare(
    inventory_path: Path = DEFAULT_INVENTORY,
    color_audit_path: Path = DEFAULT_COLOR_AUDIT,
    output_root: Path = V2_RESULT_ROOT,
    folds_path: Path = DEFAULT_FOLDS,
) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    color_audit = json.loads(color_audit_path.read_text(encoding="utf-8"))
    protocol, folds = build_protocol(inventory, color_audit)
    write_json(RESULT_ROOT / "expert_availability_audit.json", protocol["expert_availability"])
    semantics_path = RESULT_ROOT / "MASK_SEMANTICS.md"
    semantics_path.parent.mkdir(parents=True, exist_ok=True)
    semantics_path.write_text(_semantics_markdown(color_audit, protocol["mask_policies"]), encoding="utf-8")
    write_json(output_root / "protocol.json", protocol)
    write_json(output_root / "cracks" / "archive_audit" / "expert_availability_audit.json", protocol["expert_availability"])
    write_json(output_root / "cracks" / "mask_semantics" / "policies.json", protocol["mask_policies"])
    write_json(folds_path, folds)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--color-audit", type=Path, default=DEFAULT_COLOR_AUDIT)
    parser.add_argument("--output-root", type=Path, default=V2_RESULT_ROOT)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    args = parser.parse_args()
    protocol = prepare(args.inventory, args.color_audit, args.output_root, args.folds)
    print("ANZA V2 CRACKS PROTOCOL: FROZEN")
    print(f"PROTOCOL SHA256: {protocol['sha256']}")
    print(f"SETTING A TRAIN ANNOTATORS: {len(protocol['setting_a']['training_annotators'])}")
    print(f"SETTING A HELD OUT: {','.join(protocol['setting_a']['held_out_annotators']['all'])}")
    print("SETTING B FOLDS: 5 x (28 train / 4 validation / 8 test)")
    print("EXPERT SCORES: LOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
