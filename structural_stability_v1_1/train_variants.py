"""Future-training factories frozen in SS1.5; this module performs no training."""

from __future__ import annotations

from pathlib import Path

from structural_stability_v1_1.geometry_metric import V11StructuralModel
from structural_stability_v1_1.initialization import initialize_variant
from structural_stability_v1_1.protocol import SEEDS, VARIANTS


def build_fresh_variant(variant: str, seed: int, initialization_root: Path) -> V11StructuralModel:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("variant/seed is outside the frozen 12-job matrix")
    return initialize_variant(variant, seed, initialization_root / f"backbone_init_s{seed}.pt")


def assert_training_inputs(*, section_ids: list[int], expert_accessed: bool, development_accessed: bool, confirm_accessed: bool) -> None:
    if len(section_ids) != 220 or len(set(section_ids)) != 220:
        raise PermissionError("training must use exactly frozen SS_TRAIN")
    if expert_accessed or development_accessed or confirm_accessed:
        raise PermissionError("forbidden evaluation data access before V1.1 training freeze")
