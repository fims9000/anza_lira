"""Independent ANZA-KIR streams and base-only hard-candidate generation."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom

from anza_ks.benchmark.matched_generator import TASKS, _generate as generate_k1_pair
from anza_ks_k2.benchmark import SIZE, _background, _draw_curve, _package


VERSION = "ANZA_KIR_IR0_IR2_V1"
BASE_PRETRAIN_SIZE = 4096
NATURAL_SIZES = {"residual-train-natural": 2048, "calibration-natural": 512, "dev-natural": 1024, "confirm-natural": 2048}
POOL_SIZES = {"mine-train": 30_000, "mine-calibration": 5_000, "mine-dev": 10_000, "mine-confirm": 5_000}
SEEDS = {
    "base-pretrain": 3_101_000_000,
    "residual-train-natural": 3_111_000_000,
    "calibration-natural": 3_121_000_000,
    "dev-natural": 3_131_000_000,
    "confirm-natural": 3_141_000_000,
    "mine-train": 3_151_000_000,
    "mine-calibration": 3_161_000_000,
    "mine-dev": 3_171_000_000,
    "mine-confirm": 3_181_000_000,
}


def _rng(stream: str, index: int) -> np.random.Generator:
    if stream not in SEEDS:
        raise ValueError(f"unknown ANZA-KIR stream: {stream}")
    return np.random.default_rng(SEEDS[stream] + int(index))


def _new_natural(stream: str, index: int) -> dict[str, Any]:
    # This mirrors K2 scene semantics but uses disjoint deterministic seeds.
    rng = _rng(stream, index)
    target = np.zeros((SIZE, SIZE), dtype=np.float64)
    distractor = np.zeros_like(target)
    bank = np.zeros((8, SIZE, SIZE), dtype=np.float64)
    x = np.linspace(4, SIZE - 5, 180)
    for _ in range(int(rng.integers(1, 4))):
        center = rng.uniform(18, 78); amplitude = rng.uniform(2.0, 14.0)
        frequency = rng.uniform(0.65, 1.95); phase = rng.uniform(-np.pi, np.pi); slope = rng.uniform(-0.24, 0.24)
        y = center + slope * (x - SIZE / 2) + amplitude * np.sin(frequency * np.pi * x / SIZE + phase)
        visible = np.ones(len(x), dtype=bool)
        if rng.random() < 0.70:
            start = int(rng.integers(45, 120)); visible[start : start + int(rng.integers(8, 26))] = False
        _draw_curve(target, bank, x, y, visible=visible)
        if rng.random() < 0.80:
            offset = rng.choice((-1.0, 1.0)) * rng.uniform(2.5, 8.0)
            dummy = np.zeros_like(bank)
            _draw_curve(distractor, dummy, x, y + offset, visible=rng.random(len(x)) > rng.uniform(0.15, 0.42))
    target_blur = gaussian_filter(target, 0.75); target_mask = target_blur >= 0.16
    distractor_blur = gaussian_filter(distractor, 0.75); distractor_mask = (distractor_blur >= 0.16) & ~target_mask
    for mode in range(8):
        bank[mode] = gaussian_filter(bank[mode], 0.9)
    bank /= np.maximum(bank.max(axis=0, keepdims=True), 1.0)
    attenuation = gaussian_filter(rng.uniform(0.30, 1.0, size=(SIZE, SIZE)), sigma=8.0)
    image = _background(rng) + rng.uniform(0.62, 1.0) * target_blur * attenuation + rng.uniform(0.30, 0.82) * distractor_blur
    if rng.random() < 0.60:
        image = gaussian_filter(image, rng.uniform(0.35, 1.0))
    image += rng.normal(0.0, rng.uniform(0.025, 0.085), image.shape)
    return _package(image, target_mask, distractor_mask, bank, "natural", stream, index)


def _mechanism_scene(stream: str, index: int) -> dict[str, Any]:
    rng = _rng(stream, index)
    task = TASKS[index % len(TASKS)]
    source_split = "train" if stream == "base-pretrain" else "dev"
    source_size = 2048 if source_split == "train" else 1024
    source_index = int((index * 37 + TASKS.index(task) * 101) % source_size)
    pair = generate_k1_pair(task, source_split, source_index)
    positive = zoom(pair["positive"], 2.0, order=1); negative = zoom(pair["negative"], 2.0, order=1)
    angle = float(rng.choice(np.arange(8) * 22.5))
    positive = rotate(positive, angle, reshape=False, order=1, mode="reflect", prefilter=False)
    negative = rotate(negative, angle, reshape=False, order=1, mode="reflect", prefilter=False)

    def normalize(value: np.ndarray) -> np.ndarray:
        value = value - np.quantile(value, 0.15)
        return np.clip(value / (np.quantile(value, 0.99) + 1e-8), 0.0, 1.0)

    positive = normalize(positive); negative = normalize(negative)
    image = _background(rng); target = np.zeros((SIZE, SIZE), dtype=bool); distractor = np.zeros_like(target)
    bank = np.zeros((8, SIZE, SIZE), dtype=np.float64)
    positions = [(int(rng.integers(7, 20)), int(rng.integers(7, 20))), (int(rng.integers(58, 63)), int(rng.integers(58, 63)))]
    if rng.random() < 0.5:
        positions.reverse()
    if stream == "base-pretrain":
        positive_gain = rng.uniform(0.84, 1.04)
        negative_gain = rng.uniform(0.74, 1.00)
    else:
        # Base-only V1 construction produced PairError=0.049 after fixed bottom-20%
        # mining. Remove the incidental contrast shortcut before any R0--R3 model
        # exists; the percentile, tasks, base checkpoint, and gates stay frozen.
        positive_gain = rng.uniform(0.80, 1.00)
        negative_gain = rng.uniform(0.81, 1.05)
    for value, gain, (y0, x0), is_target in ((positive, positive_gain, positions[0], True), (negative, negative_gain, positions[1], False)):
        height, width = value.shape
        image[y0 : y0 + height, x0 : x0 + width] += gain * value
        local = value >= max(0.34, float(np.quantile(value, 0.72)))
        if is_target:
            target[y0 : y0 + height, x0 : x0 + width] |= local
            mode = int(round((angle % 180) / 22.5)) % 8
            bank[mode, y0 : y0 + height, x0 : x0 + width] = np.maximum(
                bank[mode, y0 : y0 + height, x0 : x0 + width], gaussian_filter(local.astype(float), 0.8)
            )
        else:
            distractor[y0 : y0 + height, x0 : x0 + width] |= local
    distractor &= ~target
    image = gaussian_filter(image, rng.uniform(0.25, 0.75)) + rng.normal(0.0, rng.uniform(0.025, 0.07), image.shape)
    sample = _package(image, target, distractor, bank, task, stream, index)
    sample["mechanism_task"] = task
    sample["positive_gain"] = float(positive_gain); sample["negative_gain"] = float(negative_gain)
    return sample


@lru_cache(maxsize=256)
def _cached(stream: str, index: int) -> dict[str, Any]:
    if stream == "base-pretrain":
        return _new_natural(stream, index) if index % 2 == 0 else _mechanism_scene(stream, index)
    if stream in NATURAL_SIZES:
        return _new_natural(stream, index)
    if stream in POOL_SIZES:
        return _mechanism_scene(stream, index)
    raise ValueError(f"unknown stream: {stream}")


def generate_sample(stream: str, index: int, *, allow_confirm: bool = False) -> dict[str, Any]:
    limit = BASE_PRETRAIN_SIZE if stream == "base-pretrain" else NATURAL_SIZES.get(stream, POOL_SIZES.get(stream, -1))
    if not 0 <= index < limit:
        raise ValueError("ANZA-KIR stream index out of range")
    if stream in ("confirm-natural", "mine-confirm") and not allow_confirm:
        raise PermissionError("ANZA-KIR confirm streams remain hash-only")
    return _cached(stream, index)


def update_digest(digest: Any, sample: dict[str, Any]) -> None:
    for key in ("image", "target", "distractor", "orientation_bank", "orientation_valid"):
        digest.update(key.encode()); digest.update(np.asarray(sample[key]).tobytes())
    digest.update(str(sample["domain"]).encode())


def selected_hash(stream: str, indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, "little")); update_digest(digest, generate_sample(stream, index, allow_confirm=True))
    return digest.hexdigest()
