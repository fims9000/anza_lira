"""S0 figures generated only from frozen geometry/results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .protocol import METHODS
from .synthetic3d.families import generate_batch
from .transport.core import initial_covariance, propagate_covariance


def _save(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(root / f"{name}.png", dpi=180); fig.savefig(root / f"{name}.svg"); plt.close(fig)


def _ellipse(covariance: np.ndarray) -> tuple[float, float, float]:
    values, vectors = np.linalg.eigh(covariance); order = np.argsort(values)[::-1]; values = values[order]; vectors = vectors[:, order]
    return 2 * np.sqrt(values[0]), 2 * np.sqrt(values[1]), float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))


def generate_figures(root: Path, fitted: dict[str, Any], per_stratum: list[dict], iid: dict, ood: dict, selective: list[dict]) -> None:
    batch = generate_batch("geom_dev_ood", 9, 1); points = batch.true_points[0]; candidates = batch.candidate_points[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3)); center = 8
    axes[0].scatter(candidates[center, :4, 0], candidates[center, :4, 1], c=["green", "red", "orange", "purple"]); axes[0].set_title("Matched center candidates")
    axes[1].plot(points[:, 0], points[:, 1], "g-o", ms=2); axes[1].plot(candidates[:, 1, 0], candidates[:, 1, 1], "r--"); axes[1].set_title("Adjacent history reveals lineage")
    _save(fig, root, "01_center_collinear_ambiguity")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3)); theta = np.asarray([0.4]);
    for axis, method in zip(axes, METHODS[1:], strict=True):
        params = fitted[method]["params"]; covariance = initial_covariance(method, theta, params)
        for step, color in zip((1, 3, 7), ("#4c78a8", "#f58518", "#e45756"), strict=True):
            local = covariance.copy()
            for _ in range(step):
                local = propagate_covariance(method, local, theta, theta, params)
            width, height, angle = _ellipse(local[0]); axis.add_patch(Ellipse((0, 0), width, height, angle=angle, fill=False, color=color, label=str(step)))
        axis.set_xlim(-8, 8); axis.set_ylim(-8, 8); axis.set_aspect("equal"); axis.set_title(method); axis.legend(title="steps")
    _save(fig, root, "02_covariance_ellipses")

    for name, index, title in (("03_close_parallel_candidates", 1, "Close-parallel candidates"), ("04_rotating_strike_composition", 0, "Rotating-strike trajectory")):
        sample = generate_batch("geom_dev_ood", index, 1); fig, axis = plt.subplots(figsize=(5, 4))
        axis.plot(sample.true_points[0, :, 0], sample.true_points[0, :, 1], "g-o", label="true")
        for candidate, color in zip((1, 2, 3), ("r", "orange", "purple"), strict=True): axis.plot(sample.candidate_points[0, :, candidate, 0], sample.candidate_points[0, :, candidate, 1], "--", color=color, alpha=.7)
        axis.set_title(title); axis.legend(); _save(fig, root, name)

    selected = [row for row in per_stratum if row["split"] == "geom_dev_ood"]
    families = sorted({row["family"] for row in selected}); x = np.arange(len(families)); fig, axis = plt.subplots(figsize=(10, 4))
    for offset, method in enumerate(METHODS):
        values = [next(row["top1"] for row in selected if row["family"] == family and row["method"] == method) for family in families]
        axis.bar(x + (offset - 2) * .15, values, width=.15, label=method)
    axis.set_xticks(x, families, rotation=30, ha="right"); axis.set_ylabel("Top1"); axis.legend(fontsize=7); _save(fig, root, "05_top1_by_stratum")

    gap_names = ["multi_slice_gap_3", "multi_slice_gap_7"]; fig, axis = plt.subplots(figsize=(6, 4));
    for method in METHODS:
        values = [next(row["switch"] for row in selected if row["family"] == family and row["method"] == method) for family in gap_names]
        axis.plot(gap_names, values, marker="o", label=method)
    axis.set_ylabel("Switch rate"); axis.legend(fontsize=7); _save(fig, root, "06_switch_by_gap")

    fig, axis = plt.subplots(figsize=(6, 4))
    for method in METHODS:
        rows = [row for row in selective if row["split"] == "geom_dev_ood" and row["method"] == method]
        axis.plot([row["coverage"] for row in rows], [row["risk"] for row in rows], label=method)
    axis.set_xlabel("Coverage"); axis.set_ylabel("Switch risk"); axis.legend(fontsize=7); _save(fig, root, "07_risk_coverage")

    fig, axis = plt.subplots(figsize=(8, 4)); x = np.arange(len(METHODS)); width=.35
    axis.bar(x-width/2, [iid[m]["top1"] for m in METHODS], width, label="IID"); axis.bar(x+width/2, [ood[m]["top1"] for m in METHODS], width, label="OOD")
    axis.set_xticks(x, METHODS, rotation=25, ha="right"); axis.set_ylabel("Top1"); axis.legend(); _save(fig, root, "08_iid_vs_ood")
