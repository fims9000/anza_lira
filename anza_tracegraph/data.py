"""Scene-level frozen synthetic relation benchmark TRACEGRAPH_RELATION_V1."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .candidates import generate_candidates
from .protocol import PROTOCOL
from .tracelets import Endpoint, Tracelet, endpoints, tracelet_token


SIZE = 96
SCENE_TYPES = tuple(PROTOCOL["scene_types"])
SPLIT_SIZES = dict(PROTOCOL["splits"])
SPLIT_SEEDS = dict(PROTOCOL["split_seeds"])


def _rng(split: str, index: int) -> np.random.Generator:
    if split not in SPLIT_SIZES or not 0 <= index < SPLIT_SIZES[split]: raise ValueError("unknown TraceGraph split/index")
    return np.random.default_rng(SPLIT_SEEDS[split] + int(index))


def _curve(x: np.ndarray, center: float, amplitude: float, phase: float, slope: float, frequency: float) -> np.ndarray:
    return center + slope * (x - 48.0) + amplitude * np.sin(frequency * np.pi * x / SIZE + phase)


def _sample_path(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    points = np.column_stack((ys, xs)); distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = max(2, int(math.ceil(distance[-1])) + 1); target = np.linspace(0.0, distance[-1], count)
    return np.column_stack((np.interp(target, distance, ys), np.interp(target, distance, xs)))


def _deposit(field: np.ndarray, points: np.ndarray, value: float = 1.0) -> None:
    integer = np.rint(points).astype(int); valid = (integer[:, 0] >= 1) & (integer[:, 0] < SIZE - 1) & (integer[:, 1] >= 1) & (integer[:, 1] < SIZE - 1)
    field[integer[valid, 0], integer[valid, 1]] = np.maximum(field[integer[valid, 0], integer[valid, 1]], value)


def _background(rng: np.random.Generator) -> np.ndarray:
    fine = gaussian_filter(rng.normal(size=(SIZE, SIZE)), (0.8, 2.0)); broad = gaussian_filter(rng.normal(size=(SIZE, SIZE)), (5.0, 11.0))
    y, x = np.mgrid[:SIZE, :SIZE]; layers = np.sin(y * rng.uniform(0.14, 0.27) + 0.7 * np.sin(x * rng.uniform(0.02, 0.055)))
    return 0.12 * fine + 0.08 * broad + 0.10 * layers


def _render_geometry(tracelets: list[Tracelet], rng: np.random.Generator, *, true_tracelet: int | None, scene_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    visible = np.zeros((SIZE, SIZE), dtype=float); c2 = np.zeros_like(visible); s2 = np.zeros_like(visible); weights = np.zeros_like(visible)
    for tracelet in tracelets:
        _deposit(visible, tracelet.points_yx)
        diffs = np.gradient(tracelet.points_yx, axis=0); angles = np.arctan2(diffs[:, 0], diffs[:, 1]); integer = np.rint(tracelet.points_yx).astype(int)
        valid = (integer[:, 0] >= 1) & (integer[:, 0] < SIZE - 1) & (integer[:, 1] >= 1) & (integer[:, 1] < SIZE - 1)
        np.add.at(c2, (integer[valid, 0], integer[valid, 1]), np.cos(2 * angles[valid])); np.add.at(s2, (integer[valid, 0], integer[valid, 1]), np.sin(2 * angles[valid])); np.add.at(weights, (integer[valid, 0], integer[valid, 1]), 1.0)
    pc = gaussian_filter(visible, 0.7); pf = gaussian_filter(visible, 1.05); pf /= max(float(pf.max()), 1e-8)
    c2 = gaussian_filter(c2, 1.2); s2 = gaussian_filter(s2, 1.2); theta = 0.5 * np.arctan2(s2, c2); confidence = np.clip(gaussian_filter(weights, 1.4), 0.0, 1.0)
    image = _background(rng) + rng.uniform(0.62, 0.95) * pf
    if true_tracelet is not None:
        source = tracelets[0].points_yx[-1]; destination = tracelets[true_tracelet].points_yx[0]; alpha = np.linspace(0, 1, 80); bridge = source[None] * (1 - alpha[:, None]) + destination[None] * alpha[:, None]
        latent = np.zeros_like(visible); _deposit(latent, bridge)
        strength = 0.12 if scene_type in ("low_contrast", "partial_occlusion") else 0.22
        image += strength * gaussian_filter(latent, 1.0)
    if scene_type == "cluttered_corridor":
        image += 0.15 * gaussian_filter(rng.normal(size=image.shape), (0.5, 2.8))
    if scene_type == "partial_occlusion": image[:, 41:53] *= 0.55
    image = gaussian_filter(image, rng.uniform(0.2, 0.65)); image += rng.normal(0.0, rng.uniform(0.025, 0.07), image.shape)
    image = (image - image.mean()) / (image.std() + 1e-8); gy, gx = np.gradient(image)
    return np.stack((image, gx, gy)).astype(np.float32), pf.astype(np.float32), pc.astype(np.float32), theta.astype(np.float32), confidence.astype(np.float32)


def _tracelet(tracelet_id: int, points: np.ndarray, probability: float, contrast: float) -> Tracelet:
    return Tracelet(tracelet_id, np.asarray(points, dtype=np.float64), probability, contrast)


def _make_scene(split: str, index: int) -> dict[str, Any]:
    rng = _rng(split, index); scene_type = SCENE_TYPES[index % len(SCENE_TYPES)]; positive = ((index // len(SCENE_TYPES)) % 2) == 0
    center = rng.uniform(29, 67); amplitude = rng.uniform(0.5, 6.5); phase = rng.uniform(-math.pi, math.pi); slope = rng.uniform(-0.18, 0.18); frequency = rng.uniform(0.65, 1.5)
    if scene_type == "straight": amplitude = 0.2
    if scene_type == "s_curve": frequency = 2.1; amplitude = rng.uniform(4.0, 8.0)
    x_source = np.linspace(4, 34, 90); y_source = _curve(x_source, center, amplitude, phase, slope, frequency); source_path = _sample_path(x_source, y_source)
    tracelets = [_tracelet(0, source_path, rng.uniform(0.72, 0.96), rng.uniform(0.6, 1.0))]
    candidate_count = int(rng.integers(4, 9)); true_id: int | None = None
    gap_end = rng.uniform(50, 64) if scene_type != "long_gap" else rng.uniform(62, 70)
    true_y = float(_curve(np.asarray([gap_end]), center, amplitude, phase, slope, frequency)[0])
    if positive:
        xs = np.linspace(gap_end, 92, 90); ys = _curve(xs, center, amplitude, phase, slope, frequency)
        tracelets.append(_tracelet(1, _sample_path(xs, ys), rng.uniform(0.65, 0.94), rng.uniform(0.55, 1.0))); true_id = 1
    while len(tracelets) < candidate_count + 1:
        trace_id = len(tracelets); offset_pool = [-15, -10, -6, 5, 9, 14, 19, -20]; offset = offset_pool[(trace_id + index) % len(offset_pool)] + rng.uniform(-1.5, 1.5)
        if scene_type in ("close_parallel", "parallel_gap_confuser"): offset = rng.choice((-1, 1)) * rng.uniform(3.0, 6.0)
        if scene_type in ("x_crossing", "acute_crossing"): offset = rng.uniform(-8, 8)
        start_x = gap_end + rng.uniform(-2.5, 5.0); xs = np.linspace(start_x, 92, 80)
        local_slope = slope + rng.uniform(-0.25, 0.25)
        if scene_type == "x_crossing": local_slope += rng.choice((-1, 1)) * rng.uniform(0.45, 0.75)
        if scene_type == "acute_crossing": local_slope += rng.choice((-1, 1)) * rng.uniform(0.25, 0.42)
        ys = true_y + offset + local_slope * (xs - start_x) + rng.uniform(0.0, 2.0) * np.sin(rng.uniform(0.7, 1.4) * np.pi * xs / SIZE + rng.uniform(-math.pi, math.pi))
        if np.all((ys > 3) & (ys < SIZE - 4)): tracelets.append(_tracelet(trace_id, _sample_path(xs, ys), rng.uniform(0.50, 0.92), rng.uniform(0.45, 1.0)))
    image, pf, pc, theta, confidence = _render_geometry(tracelets, rng, true_tracelet=true_id, scene_type=scene_type)
    source_endpoint = endpoints(tracelets[0], 5)[1]
    destination_endpoints = [endpoints(tracelet, 5)[0] for tracelet in tracelets[1:]]
    candidate_settings = PROTOCOL["candidates"]
    candidates = generate_candidates(source_endpoint, destination_endpoints, k_max=int(candidate_settings["k_max"]), min_distance=float(candidate_settings["min_distance"]), max_distance=float(candidate_settings["max_distance"]), max_tangent_error=math.radians(float(candidate_settings["max_tangent_mismatch_degrees"])))
    candidate_ids = [candidate.endpoint.tracelet_id for candidate in candidates]
    target_index = candidate_ids.index(true_id) if true_id in candidate_ids else -1
    tokens = [tracelet_token(tracelet, pf, image[0], (SIZE, SIZE)) for tracelet in tracelets]
    dense = np.concatenate((image, pf[None], pc[None], np.cos(2 * theta)[None], np.sin(2 * theta)[None], confidence[None]), axis=0).astype(np.float32)
    return {"dense": dense, "tracelets": tracelets, "tracelet_tokens": np.stack(tokens), "source_endpoint": source_endpoint, "candidates": candidates, "candidate_ids": candidate_ids, "target_index": target_index, "has_valid_continuation": positive, "candidate_recalled": bool(not positive or target_index >= 0), "scene_type": scene_type, "split": split, "index": index}


def generate_scene(split: str, index: int, *, allow_confirm: bool = False) -> dict[str, Any]:
    if split == "confirm" and not allow_confirm: raise PermissionError("TraceGraph confirm is hash-only through TG2")
    return _make_scene(split, index)


def scene_digest(scene: dict[str, Any]) -> bytes:
    digest = hashlib.sha256(); digest.update(scene["dense"].tobytes()); digest.update(str(scene["scene_type"]).encode()); digest.update(int(scene["target_index"]).to_bytes(2, "little", signed=True))
    for tracelet in scene["tracelets"]: digest.update(tracelet.points_yx.tobytes())
    digest.update(np.asarray(scene["candidate_ids"], dtype=np.int16).tobytes()); return digest.digest()


def split_hash(split: str) -> str:
    digest = hashlib.sha256()
    for index in range(SPLIT_SIZES[split]): digest.update(index.to_bytes(4, "little")); digest.update(scene_digest(generate_scene(split, index, allow_confirm=True)))
    return digest.hexdigest()
