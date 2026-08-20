"""Mechanism figures generated from frozen ANZA-S oracle trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def _paths(rows: list[dict[str, Any]], *, index: int, task: str, pair_id: str, method: str) -> list[list[dict[str, Any]]]:
    selected = [row for row in rows if int(row["index"]) == index and row["task"] == task and row["pair_id"] == pair_id and row["method"] == method]
    return [sorted([row for row in selected if row["side"] == side], key=lambda row: int(row["step"])) for side in ("left", "right")]


def _base(axis, index: int, title: str) -> None:
    sample = generate_sample_v4("validation", index, image_size=64)
    image = np.asarray(sample["image"])
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.mean(axis=0)
    axis.imshow(image, cmap="gray", origin="upper")
    axis.set_title(title, fontsize=9); axis.set_xticks([]); axis.set_yticks([])


def _draw(axis, paths, colors=("#00d4ff", "#ffcc00")) -> None:
    for path, color in zip(paths, colors, strict=True):
        axis.plot([row["x"] for row in path], [row["y"] for row in path], "o-", color=color, lw=2, ms=3)
        for row in path:
            axis.quiver(row["x"], row["y"], row["ux"], row["uy"], color=color, angles="xy", scale_units="xy", scale=0.55, width=0.006)


def _save(fig, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def generate_figures(root: Path, trajectories: list[dict[str, Any]], scores: list[dict[str, Any]]) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    x_index = min(int(row["index"]) for row in scores if row["case"] == "x_junction")
    x_correct = next(row for row in scores if int(row["index"]) == x_index and row["task"] == "x_correct" and row["method"] == "O4_cocycle_shadowing")
    x_wrong = next(row for row in scores if int(row["index"]) == x_index and row["task"] == "x_wrong_turn" and row["method"] == "O4_cocycle_shadowing")
    o0 = next(row for row in scores if int(row["index"]) == x_index and row["task"] == "x_correct" and row["pair_id"] == x_correct["pair_id"] and row["method"] == "O0_scalar_anza")
    o1 = next(row for row in scores if int(row["index"]) == x_index and row["task"] == "x_correct" and row["pair_id"] == x_correct["pair_id"] and row["method"] == "O1_mode_state")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    _base(axes[0], x_index, f"O0 scalar score={o0['score']:.3f}")
    _base(axes[1], x_index, f"O1 mode-state score={o1['score']:.3f}")
    _base(axes[2], x_index, "O3 cocycle: correct")
    _draw(axes[2], _paths(trajectories, index=x_index, task="x_correct", pair_id=x_correct["pair_id"], method="O3_cocycle_rollout"))
    _base(axes[3], x_index, "O3 cocycle: wrong turn")
    _draw(axes[3], _paths(trajectories, index=x_index, task="x_wrong_turn", pair_id=x_wrong["pair_id"], method="O3_cocycle_rollout"))
    _save(fig, root, "F1_x_scalar_state_cocycle")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for axis, row, label in ((axes[0], x_correct, "correct"), (axes[1], x_wrong, "wrong")):
        _base(axis, x_index, f"X {label}: S={row['score']:.3f}")
        _draw(axis, _paths(trajectories, index=x_index, task=row["task"], pair_id=row["pair_id"], method="O4_cocycle_shadowing"))
    _save(fig, root, "F2_x_shadowing_correct_wrong")

    parallel_index = min(int(row["index"]) for row in scores if row["case"] == "near_parallel")
    parallel = next(row for row in scores if int(row["index"]) == parallel_index and row["task"] == "parallel_wrong" and row["method"] == "O4_cocycle_shadowing")
    fig, axis = plt.subplots(figsize=(4, 4)); _base(axis, parallel_index, f"Parallel negative: S={parallel['score']:.3f}")
    _draw(axis, _paths(trajectories, index=parallel_index, task="parallel_wrong", pair_id=parallel["pair_id"], method="O4_cocycle_shadowing"))
    _save(fig, root, "F3_parallel_separation")

    curved_index = min(int(row["index"]) for row in scores if row["case"] == "curved_fault")
    curved = next(row for row in scores if int(row["index"]) == curved_index and row["task"] == "curved_gap" and row["method"] == "O4_cocycle_shadowing")
    fig, axis = plt.subplots(figsize=(4, 4)); _base(axis, curved_index, f"Derived curved gap: S={curved['score']:.3f}")
    _draw(axis, _paths(trajectories, index=curved_index, task="curved_gap", pair_id=curved["pair_id"], method="O4_cocycle_shadowing"))
    _save(fig, root, "F4_curved_gap_shadowing")
    return ["F1_x_scalar_state_cocycle", "F2_x_shadowing_correct_wrong", "F3_parallel_separation", "F4_curved_gap_shadowing"]
