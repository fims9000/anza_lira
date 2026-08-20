"""Dedicated surface-family constructors with immutable lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..protocol import FAMILIES, SPLITS


STEPS = 17
CANDIDATES = 5
END_INDEX = 4


@dataclass(frozen=True)
class FamilySpec:
    name: str
    rotation: float = 0.0
    curve: float = 0.03
    s_warp: float = 0.0
    parallel_spacing: float = 4.0
    spacing_slope: float = 0.0
    gap: int = 0
    stepover: float = 0.0
    crossing: bool = False
    collinear: bool = False
    competing: bool = False
    terminate_at: int = -1
    near_touch: bool = False


def make_plane_smooth() -> FamilySpec: return FamilySpec("plane_smooth", curve=0.005)
def make_curved_surface() -> FamilySpec: return FamilySpec("curved_surface", curve=0.075)
def make_rotating_strike() -> FamilySpec: return FamilySpec("rotating_strike", rotation=0.055, curve=0.05)
def make_s_warp() -> FamilySpec: return FamilySpec("s_warp", curve=0.035, s_warp=1.2)
def make_stepover() -> FamilySpec: return FamilySpec("stepover", curve=0.04, stepover=2.5)
def make_close_parallel() -> FamilySpec: return FamilySpec("close_parallel", parallel_spacing=1.7, curve=0.04)
def make_diverging_parallel() -> FamilySpec: return FamilySpec("diverging_parallel", parallel_spacing=1.3, spacing_slope=0.30)
def make_converging_parallel() -> FamilySpec: return FamilySpec("converging_parallel", parallel_spacing=5.2, spacing_slope=-0.28)
def make_projection_crossing() -> FamilySpec: return FamilySpec("projection_crossing", crossing=True, curve=0.04)
def make_center_collinear_ambiguous() -> FamilySpec: return FamilySpec("center_collinear_ambiguous", collinear=True, curve=0.05)
def make_multi_slice_gap_1() -> FamilySpec: return FamilySpec("multi_slice_gap_1", gap=1, curve=0.05)
def make_multi_slice_gap_3() -> FamilySpec: return FamilySpec("multi_slice_gap_3", gap=3, curve=0.06)
def make_multi_slice_gap_7() -> FamilySpec: return FamilySpec("multi_slice_gap_7", gap=7, curve=0.075)
def make_competing_branch() -> FamilySpec: return FamilySpec("competing_branch", competing=True, parallel_spacing=2.3, curve=0.055)
def make_terminating_surface() -> FamilySpec: return FamilySpec("terminating_surface", terminate_at=12, curve=0.04)
def make_two_surface_near_touch() -> FamilySpec: return FamilySpec("two_surface_near_touch", near_touch=True, curve=0.05)
def make_combined_rotate_gap_parallel() -> FamilySpec: return FamilySpec("combined_rotate_gap_parallel", rotation=0.075, curve=0.09, gap=3, parallel_spacing=1.25)


CONSTRUCTORS: dict[str, Callable[[], FamilySpec]] = {
    name: globals()[f"make_{name}"] for name in FAMILIES
}


@dataclass
class CaseBatch:
    scene_id: np.ndarray
    family: np.ndarray
    surface_ids: np.ndarray
    true_points: np.ndarray
    true_theta: np.ndarray
    candidate_points: np.ndarray
    candidate_theta: np.ndarray
    candidate_valid: np.ndarray
    observed: np.ndarray
    truth_index: np.ndarray


def _angles(spec: FamilySpec, rng: np.random.Generator, *, ood: bool) -> np.ndarray:
    z = np.arange(STEPS, dtype=np.float64) - 8.0
    base = rng.uniform(0.15, np.pi - 0.15)
    multiplier = 1.45 if ood else 1.0
    return np.mod(base + multiplier * spec.rotation * z + 0.025 * spec.s_warp * np.sin(np.pi * z / 8), np.pi)


def _one_case(name: str, scene_id: int, seed: int, *, ood: bool) -> tuple:
    spec = CONSTRUCTORS[name](); rng = np.random.default_rng(seed)
    theta = _angles(spec, rng, ood=ood); u = np.stack([np.cos(theta), np.sin(theta)], axis=-1); s = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    points = np.empty((STEPS, 2), dtype=np.float64); points[0] = rng.uniform(36, 60, size=2)
    velocity = rng.normal(0, 0.28, size=2); multiplier = 1.55 if ood else 1.0
    for k in range(1, STEPS):
        phase = np.sin((k - 1) * np.pi / 7 + rng.uniform(-0.05, 0.05))
        acceleration = multiplier * spec.curve * phase * u[k - 1] + 0.18 * spec.curve * np.cos(k) * s[k - 1]
        if spec.s_warp:
            acceleration += 0.025 * spec.s_warp * np.cos(k * np.pi / 4) * u[k - 1]
        velocity = velocity + acceleration; points[k] = points[k - 1] + velocity
    if spec.stepover:
        points[8:] += spec.stepover * s[8]

    candidates = np.empty((STEPS, CANDIDATES, 2), dtype=np.float64)
    candidate_theta = np.empty((STEPS, CANDIDATES), dtype=np.float64)
    localization = rng.normal(0, 0.08, size=(STEPS, 2))
    candidates[:, 0] = points + localization; candidate_theta[:, 0] = np.mod(theta + rng.normal(0, 0.006, STEPS), np.pi)
    z = np.arange(STEPS, dtype=np.float64) - 8.0
    spacing = spec.parallel_spacing + spec.spacing_slope * z
    if ood:
        spacing *= 0.72
    if spec.near_touch:
        spacing = 0.65 + 0.45 * np.abs(z)
    candidates[:, 1] = points + spacing[:, None] * s; candidate_theta[:, 1] = theta
    candidates[:, 2] = points + (1.4 + 0.10 * z)[:, None] * u + 0.55 * s; candidate_theta[:, 2] = theta
    cross_direction = np.stack([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)], axis=-1)
    candidates[:, 3] = points + (0.55 * z)[:, None] * cross_direction
    candidate_theta[:, 3] = np.mod(theta + (np.pi / 2 if spec.crossing else 0.18), np.pi)
    if spec.collinear:
        candidates[:, 1] = points + (0.38 * np.abs(z))[:, None] * s
        candidates[8, 1] = candidates[8, 0]
        candidate_theta[:, 1] = theta
    if spec.competing:
        candidates[:, 2] = points + (0.75 + 0.07 * np.abs(z))[:, None] * s + 0.35 * u
    candidates[:, END_INDEX] = np.nan; candidate_theta[:, END_INDEX] = theta

    observed = np.ones(STEPS, dtype=bool); observed[0] = True
    if spec.gap:
        start = 8 - spec.gap // 2; observed[start:start + spec.gap] = False
    valid = np.ones((STEPS, CANDIDATES), dtype=bool); valid[:, END_INDEX] = False
    truth = np.zeros(STEPS, dtype=np.int8)
    if spec.terminate_at >= 0:
        valid[spec.terminate_at:, :END_INDEX] = False; valid[spec.terminate_at:, END_INDEX] = True
        truth[spec.terminate_at:] = END_INDEX
    surface_ids = np.asarray([scene_id * 10 + index for index in range(CANDIDATES)], dtype=np.int64)
    return surface_ids, points, theta, candidates, candidate_theta, valid, observed, truth


def generate_batch(split_name: str, start: int, count: int) -> CaseBatch:
    if split_name not in SPLITS or split_name == "geom_confirm":
        raise PermissionError("confirm is hash-only and cannot be generated in S0")
    split = SPLITS[split_name]; end = min(int(split["size"]), int(start) + int(count))
    indices = np.arange(int(start), end, dtype=np.int64); ood = split_name == "geom_dev_ood"
    available = list(FAMILIES if not ood else (
        "rotating_strike", "close_parallel", "multi_slice_gap_3", "multi_slice_gap_7",
        "center_collinear_ambiguous", "combined_rotate_gap_parallel",
    ))
    payload = []
    for index in indices:
        name = available[int(index) % len(available)]
        seed = int(split["seed"]) * 1_000_003 + int(index)
        payload.append((name, _one_case(name, int(index) + int(split["seed"]) * 100_000, seed, ood=ood)))
    return CaseBatch(
        scene_id=indices + int(split["seed"]) * 100_000,
        family=np.asarray([item[0] for item in payload]),
        surface_ids=np.stack([item[1][0] for item in payload]),
        true_points=np.stack([item[1][1] for item in payload]),
        true_theta=np.stack([item[1][2] for item in payload]),
        candidate_points=np.stack([item[1][3] for item in payload]),
        candidate_theta=np.stack([item[1][4] for item in payload]),
        candidate_valid=np.stack([item[1][5] for item in payload]),
        observed=np.stack([item[1][6] for item in payload]),
        truth_index=np.stack([item[1][7] for item in payload]),
    )


def observability_dataset(split_name: str, count: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matched center descriptors plus adjacent-history oracle scores."""
    if split_name == "geom_confirm":
        raise PermissionError("confirm locked")
    seed = int(SPLITS[split_name]["seed"]) + 991
    rng = np.random.default_rng(seed); labels = np.arange(count) % 2
    latent = rng.normal(size=(count, 4))
    # Center descriptors are generated independently of lineage label.
    center = np.column_stack([
        np.abs(latent[:, 0]) + 3.0, np.mod(np.abs(latent[:, 1]), 0.2),
        np.abs(latent[:, 2]) * 0.04, np.abs(latent[:, 3]) * 4 + 12,
    ])
    # Adjacent history is consistent for same-surface examples and inconsistent otherwise.
    context_score = rng.normal(1.0, 0.08, count) - labels * 0.0
    context_score[labels == 0] = rng.normal(-1.0, 0.08, np.count_nonzero(labels == 0))
    return center.astype(np.float64), labels.astype(np.int8), context_score.astype(np.float64)
