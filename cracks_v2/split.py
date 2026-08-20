"""Deterministic grouped OOF split without invented spatial coordinates."""

from __future__ import annotations

from typing import Iterable, Any


def _near_any(value: int, references: set[int], radius: int) -> bool:
    return any(abs(int(value) - int(reference)) <= int(radius) for reference in references)


def build_grouped_oof_split(
    section_ids: Iterable[int],
    *,
    folds: int = 5,
    nominal_max_section: int = 400,
    exclusion_radius: int = 2,
    dev_sections: int = 16,
) -> dict[str, Any]:
    """Group by contiguous numeric IDs while explicitly disclaiming true spatial coordinates."""

    sections = sorted(set(int(value) for value in section_ids))
    if folds < 2 or not sections:
        raise ValueError("at least two folds and one section are required")
    if nominal_max_section < max(sections):
        raise ValueError("nominal section range does not cover available IDs")
    width = (int(nominal_max_section) + folds - 1) // folds
    rows = []
    for fold in range(folds):
        lower = fold * width + 1
        upper = min((fold + 1) * width, nominal_max_section)
        outer = {value for value in sections if lower <= value <= upper}
        if not outer:
            raise ValueError(f"fold {fold} has no outer sections")
        outer_buffer = {
            value for value in sections
            if value not in outer and _near_any(value, outer, exclusion_radius)
        }
        dev_group = (fold + folds // 2) % folds
        dev_lower = dev_group * width + 1
        dev_upper = min((dev_group + 1) * width, nominal_max_section)
        dev_pool = [
            value for value in sections
            if dev_lower <= value <= dev_upper and value not in outer and value not in outer_buffer
        ]
        take = min(int(dev_sections), len(dev_pool))
        start = max(0, (len(dev_pool) - take) // 2)
        dev = set(dev_pool[start : start + take])
        dev_buffer = {
            value for value in sections
            if value not in outer and value not in dev and _near_any(value, dev, exclusion_radius)
        }
        excluded = outer | outer_buffer | dev | dev_buffer
        train = set(sections) - excluded
        if train & dev or train & outer or dev & outer:
            raise AssertionError("section overlap in grouped OOF split")
        rows.append({
            "fold": fold,
            "outer_nominal_range": [lower, upper],
            "train_section_ids": sorted(train),
            "dev_section_ids": sorted(dev),
            "outer_section_ids": sorted(outer),
            "outer_buffer_section_ids": sorted(outer_buffer),
            "dev_buffer_section_ids": sorted(dev_buffer),
            "counts": {
                "train": len(train),
                "dev": len(dev),
                "outer": len(outer),
                "outer_buffer": len(outer_buffer),
                "dev_buffer": len(dev_buffer),
            },
        })
    outer_union = set().union(*(set(row["outer_section_ids"]) for row in rows))
    return {
        "version": "anza2_cracks_grouped_oof_v1",
        "status": "PASS_GROUPED_SECTION_OOF_WITH_SPATIAL_LIMITATION",
        "fold_count": int(folds),
        "section_count": len(sections),
        "outer_union_count": len(outer_union),
        "outer_exactly_once": len(outer_union) == len(sections) and sum(
            len(row["outer_section_ids"]) for row in rows
        ) == len(sections),
        "exclusion_radius_numeric_ids": int(exclusion_radius),
        "spatial_coordinates_status": "NOT_ESTABLISHED",
        "grouping_basis": "contiguous numeric section IDs from the official archive",
        "limitation": "Numeric ordering is used for grouped OOF; physical coordinates and adjacency were not present in the release and are not inferred.",
        "expert_data_accessed": False,
        "folds": rows,
    }
