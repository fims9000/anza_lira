"""Deterministic handcrafted ANZA-2 fields for Phase-1 mechanism evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .affinity import ANZA2StructuralAffinity, LOCAL8_OFFSETS
from .field import ANZA2FieldOutput


def _field(membership: torch.Tensor, angles: torch.Tensor, *, scale: float, hyper: float) -> ANZA2FieldOutput:
    orientation = torch.stack((torch.cos(2.0 * angles), torch.sin(2.0 * angles)), dim=2)
    base = torch.full_like(membership, float(scale))
    h = torch.full_like(membership, float(hyper))
    return ANZA2FieldOutput(membership, orientation, base, h, base * torch.exp(h), base * torch.exp(-h))


def _blank(size: int = 33, modes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    membership = torch.full((1, modes, size, size), 0.01, dtype=torch.float64)
    angles = torch.zeros_like(membership)
    for mode in range(modes):
        angles[:, mode] = mode * math.pi / modes
    return membership, angles


def handcrafted_fixtures(size: int = 33) -> dict[str, dict[str, Any]]:
    center = size // 2
    affinity_module = ANZA2StructuralAffinity()
    fixtures: dict[str, dict[str, Any]] = {}

    membership, angles = _blank(size)
    membership[0, 0, center, 4:-4] = 0.98
    straight = _field(membership, angles, scale=0.75, hyper=1.25)
    fixtures["straight"] = {"field": straight, "affinity": affinity_module(straight)}

    membership, angles = _blank(size)
    for y in (center - 6, center + 6):
        membership[0, 0, y, 4:-4] = 0.98
    parallel = _field(membership, angles, scale=0.75, hyper=1.25)
    fixtures["parallel"] = {"field": parallel, "affinity": affinity_module(parallel)}

    membership, angles = _blank(size)
    membership[0, 0, center, 4:-4] = 0.98
    membership[0, 1, 4:-4, center] = 0.98
    angles[0, 1, 4:-4, center] = math.pi / 2
    crossing = _field(membership, angles, scale=0.75, hyper=1.25)
    fixtures["crossing"] = {"field": crossing, "affinity": affinity_module(crossing)}

    membership, angles = _blank(size)
    path = []
    for x in range(4, size - 4):
        relative = x - center
        y = int(round(center + 0.018 * relative * relative - 2.0))
        path.append((y, x))
    for index, (y, x) in enumerate(path):
        before = path[max(0, index - 1)]
        after = path[min(len(path) - 1, index + 1)]
        tangent = math.atan2(after[0] - before[0], after[1] - before[1])
        membership[0, 0, y, x] = 0.98
        angles[0, 0, y, x] = tangent
    curved = _field(membership, angles, scale=1.1, hyper=0.70)
    fixtures["curved"] = {"field": curved, "affinity": affinity_module(curved), "path": path}
    return fixtures


def fixture_metrics(fixtures: dict[str, dict[str, Any]]) -> dict[str, float | bool]:
    center = fixtures["straight"]["field"].membership.shape[-1] // 2
    right = LOCAL8_OFFSETS.index((1, 0))
    down = LOCAL8_OFFSETS.index((0, 1))
    straight = fixtures["straight"]["affinity"]
    parallel = fixtures["parallel"]["affinity"]
    crossing = fixtures["crossing"]["affinity"]
    parallel_y = center - 6
    curved_values = []
    curved = fixtures["curved"]
    for first, second in zip(curved["path"], curved["path"][1:]):
        dy, dx = second[0] - first[0], second[1] - first[1]
        channel = LOCAL8_OFFSETS.index((dx, dy))
        curved_values.append(float(curved["affinity"][0, channel, first[0], first[1]]))
    metrics = {
        "straight_along_affinity": float(straight[0, right, center, center]),
        "straight_cross_affinity": float(straight[0, down, center, center]),
        "parallel_along_affinity": float(parallel[0, right, parallel_y, center]),
        "parallel_cross_into_gap_affinity": float(parallel[0, down, parallel_y, center]),
        "crossing_horizontal_affinity": float(crossing[0, right, center, center]),
        "crossing_vertical_affinity": float(crossing[0, down, center, center]),
        "crossing_active_modes": int((fixtures["crossing"]["field"].membership[0, :, center, center] >= 0.5).sum()),
        "curved_min_path_affinity": min(curved_values),
        "curved_mean_path_affinity": float(np.mean(curved_values)),
    }
    metrics["phase1_fixture_gate_pass"] = bool(
        metrics["straight_along_affinity"] > 0.8
        and metrics["straight_cross_affinity"] < 0.05
        and metrics["parallel_along_affinity"] > 0.8
        and metrics["parallel_cross_into_gap_affinity"] < 0.05
        and metrics["crossing_horizontal_affinity"] > 0.8
        and metrics["crossing_vertical_affinity"] > 0.8
        and metrics["crossing_active_modes"] == 2
        and metrics["curved_min_path_affinity"] > 0.55
    )
    return metrics


def save_fixture_artifacts(output_root: Path) -> dict[str, float | bool]:
    """Persist raw arrays and a code-generated diagnostic figure."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_root.mkdir(parents=True, exist_ok=True)
    fixtures = handcrafted_fixtures()
    metrics = fixture_metrics(fixtures)
    arrays = {}
    for name, item in fixtures.items():
        arrays[f"{name}_membership"] = item["field"].membership.detach().cpu().numpy()
        arrays[f"{name}_affinity"] = item["affinity"].detach().cpu().numpy()
    np.savez_compressed(output_root / "fixture_maps.npz", **arrays)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for column, name in enumerate(("straight", "parallel", "crossing", "curved")):
        membership = fixtures[name]["field"].membership.amax(dim=1)[0].cpu().numpy()
        affinity = fixtures[name]["affinity"].amax(dim=1)[0].cpu().numpy()
        axes[0, column].imshow(membership, vmin=0, vmax=1, cmap="viridis")
        axes[0, column].set_title(f"{name}: fuzzy union")
        axes[1, column].imshow(affinity, vmin=0, vmax=1, cmap="magma")
        axes[1, column].set_title(f"{name}: max local edge")
        for row in range(2):
            axes[row, column].set_axis_off()
    fig.savefig(output_root / "geometry_fixtures.png", dpi=300)
    fig.savefig(output_root / "geometry_fixtures.svg")
    plt.close(fig)
    return metrics
