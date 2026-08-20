from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.prepare_cracks_protocol import (
    BLUE,
    GREEN,
    ORANGE,
    WHITE,
    build_expert_folds,
    build_protocol,
    mask_policies,
    select_crowd_holdout,
)


def test_holdout_is_deterministic_and_has_required_expertise() -> None:
    counts = {**{f"novice{i:02d}": 330 + i for i in range(1, 27)}, **{f"practitioner{i}": 370 + i for i in range(1, 9)}}
    first = select_crowd_holdout(counts)
    second = select_crowd_holdout(dict(reversed(list(counts.items()))))
    assert first == second
    assert len(first["novices"]) == 2
    assert len(first["practitioners"]) == 1
    assert all(first["coverage"][name] >= 300 for name in first["all"])


def test_expert_folds_are_28_4_8_and_cover_test_once() -> None:
    sections = list(range(10, 410, 10))
    folds = build_expert_folds(sections)
    assert len(folds) == 5
    for fold in folds:
        assert (len(fold["train"]), len(fold["validation"]), len(fold["test"])) == (28, 4, 8)
        assert not (set(fold["train"]) & set(fold["validation"]))
        assert not (set(fold["train"]) & set(fold["test"]))
        assert not (set(fold["validation"]) & set(fold["test"]))
    assert Counter(section for fold in folds for section in fold["test"]) == Counter(sections)


def test_mask_policies_never_silently_assign_white() -> None:
    policies = mask_policies()
    assert list(WHITE) in policies["paper_like"]["negative_rgb"]
    assert "INFERRED" in policies["paper_like"]["white_status"]
    assert list(WHITE) in policies["conservative"]["ignore_rgb"]
    assert list(BLUE) in policies["paper_like"]["positive_rgb"]
    assert list(GREEN) in policies["paper_like"]["positive_rgb"]
    assert list(ORANGE) in policies["paper_like"]["ignore_rgb"]


def test_real_protocol_has_no_expert_gradient_and_frozen_folds() -> None:
    inventory_path = Path("results/cracks_study/archive_inventory.json")
    color_path = Path("results/cracks_study/expert_color_audit.json")
    if not inventory_path.exists() or not color_path.exists():
        return
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    colors = json.loads(color_path.read_text(encoding="utf-8"))
    protocol, folds = build_protocol(inventory, colors)
    assert protocol["setting_a"]["expert_gradient"] is False
    assert "expert" not in protocol["setting_a"]["training_annotators"]
    assert protocol["expert_availability"]["available_expert_subset_sections"] == 40
    assert protocol["expert_availability"]["released_expert_test_masks_found"] is False
    assert protocol["setting_a"]["training_section_count"] == 393
    assert protocol["setting_a"]["orphan_crowd_annotation_section_ids"] == [9, 185, 249, 336]
    assert protocol["setting_a"]["images_without_any_crowd_annotation"] == [49, 73, 385]
    assert len(folds["sha256"]) == 64
    assert protocol["test_access"]["expert_scores_unlocked"] is False
