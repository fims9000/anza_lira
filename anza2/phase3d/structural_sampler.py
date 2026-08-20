"""Frozen eight-strata case sampler; no contiguous-prefix surrogate."""

from __future__ import annotations

from typing import Any

import numpy as np


MANDATORY_STRATA = {
    "StraightGap": "fault_with_gap",
    "NegativeGap": "negative_gap",
    "Curve": "curved_fault",
    "ParallelFault": "near_parallel",
    "XCrossing": "x_junction",
    "WeakBranchCrossing": "weak_branch_crossing",
    "TJunction": "t_junction",
    "YJunction": "y_junction",
}


def strata_inventory(rows: list[dict[str, Any]], *, split: str) -> dict[str, list[int]]:
    selected = [row for row in rows if row["split"] == split]
    output = {}
    for stratum, case in MANDATORY_STRATA.items():
        output[stratum] = sorted(int(row["index"]) for row in selected if row["case"] == case)
    return output


def balanced_curriculum_indices(
    rows: list[dict[str, Any]],
    *,
    split: str = "train",
    quota: int = 64,
    seed: int = 20260818,
) -> list[dict[str, int | str]]:
    if quota <= 0:
        raise ValueError("quota must be positive")
    pools = strata_inventory(rows, split=split)
    missing = [name for name, values in pools.items() if not values]
    if missing:
        raise ValueError(f"mandatory strata missing: {missing}")
    rng = np.random.default_rng(seed)
    schedule = []
    for stratum, pool in pools.items():
        chosen = rng.choice(pool, size=quota, replace=len(pool) < quota)
        schedule.extend({"stratum": stratum, "case": MANDATORY_STRATA[stratum], "index": int(index)} for index in chosen)
    rng.shuffle(schedule)
    return schedule

