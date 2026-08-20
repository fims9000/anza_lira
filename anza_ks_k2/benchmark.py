"""Frozen dense segmentation benchmark for ANZA-KS K2."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom

from anza_ks.benchmark.matched_generator import TASKS, _generate as generate_k1_pair


VERSION = "ANZA_KS_SYMBOLIC_SEG_V1"
SIZE = 96
SPLIT_SIZES = {"train": 4096, "dev-natural": 1024, "dev-mechanism": 1024, "confirm": 2048}
SPLIT_SEEDS = {"train": 2_019_410_000, "dev-natural": 2_019_420_000, "dev-mechanism": 2_019_430_000, "confirm": 2_019_440_000}


def _rng(split: str, index: int) -> np.random.Generator:
    if split not in SPLIT_SIZES or not 0 <= index < SPLIT_SIZES[split]:
        raise ValueError("unknown K2 split/index")
    return np.random.default_rng(SPLIT_SEEDS[split] + int(index))


def _background(rng: np.random.Generator) -> np.ndarray:
    fine = gaussian_filter(rng.normal(size=(SIZE, SIZE)), sigma=(0.8, 2.1))
    broad = gaussian_filter(rng.normal(size=(SIZE, SIZE)), sigma=(4.0, 10.0))
    y, x = np.mgrid[:SIZE, :SIZE]
    layers = np.sin(y * rng.uniform(0.16, 0.28) + 0.8 * np.sin(x * rng.uniform(0.025, 0.055)))
    return 0.12 * fine + 0.08 * broad + 0.10 * layers


def _draw_curve(mask: np.ndarray, bank: np.ndarray, xs: np.ndarray, ys: np.ndarray, *, visible: np.ndarray | None = None) -> None:
    dx = np.gradient(xs); dy = np.gradient(ys)
    angles = np.mod(np.arctan2(dy, dx), np.pi)
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        xi, yi = int(round(x)), int(round(y))
        if not (1 <= xi < SIZE - 1 and 1 <= yi < SIZE - 1):
            continue
        if visible is None or visible[i]:
            mask[yi, xi] = 1.0
            distances = 1.0 - np.cos(2.0 * (np.arange(8) * np.pi / 8.0 - angles[i]))
            bank[:, yi, xi] = np.maximum(bank[:, yi, xi], np.exp(-distances / (2 * 0.20**2)))


def _natural(split: str, index: int, rng: np.random.Generator) -> dict[str, Any]:
    target = np.zeros((SIZE, SIZE), dtype=np.float64)
    distractor = np.zeros_like(target)
    bank = np.zeros((8, SIZE, SIZE), dtype=np.float64)
    count = int(rng.integers(1, 4))
    x = np.linspace(4, SIZE - 5, 180)
    for branch in range(count):
        center = rng.uniform(18, 78)
        amplitude = rng.uniform(2.0, 13.0)
        frequency = rng.uniform(0.7, 1.8)
        phase = rng.uniform(-np.pi, np.pi)
        slope = rng.uniform(-0.22, 0.22)
        y = center + slope * (x - SIZE / 2) + amplitude * np.sin(frequency * np.pi * (x / SIZE) + phase)
        visible = np.ones(len(x), dtype=bool)
        if rng.random() < 0.65:
            start = int(rng.integers(55, 115)); visible[start : start + int(rng.integers(7, 24))] = False
        _draw_curve(target, bank, x, y, visible=visible)
        if rng.random() < 0.75:
            offset = rng.choice((-1.0, 1.0)) * rng.uniform(3.0, 8.0)
            distractor_visible = rng.random(len(x)) > rng.uniform(0.15, 0.40)
            dummy = np.zeros_like(bank)
            _draw_curve(distractor, dummy, x, y + offset, visible=distractor_visible)
    target = gaussian_filter(target, 0.75)
    target_mask = target >= 0.16
    distractor = gaussian_filter(distractor, 0.75)
    distractor_mask = (distractor >= 0.16) & ~target_mask
    for mode in range(8):
        bank[mode] = gaussian_filter(bank[mode], 0.9)
    bank /= np.maximum(bank.max(axis=0, keepdims=True), 1.0)
    attenuation = gaussian_filter(rng.uniform(0.35, 1.0, size=(SIZE, SIZE)), sigma=8.0)
    image = _background(rng) + rng.uniform(0.65, 1.0) * target * attenuation + rng.uniform(0.35, 0.80) * distractor
    if rng.random() < 0.55:
        image = gaussian_filter(image, rng.uniform(0.35, 0.9))
    image += rng.normal(0.0, rng.uniform(0.02, 0.08), image.shape)
    return _package(image, target_mask, distractor_mask, bank, "natural", split, index)


def _mechanism(split: str, index: int, rng: np.random.Generator) -> dict[str, Any]:
    task = TASKS[index % len(TASKS)]
    source_split = "train" if split == "train" else "dev"
    pair = generate_k1_pair(task, source_split, index % (2048 if source_split == "train" else 1024))
    positive = zoom(pair["positive"], 2.0, order=1)
    negative = zoom(pair["negative"], 2.0, order=1)
    angle = float(rng.choice(np.arange(8) * 22.5))
    positive = rotate(positive, angle, reshape=False, order=1, mode="reflect", prefilter=False)
    negative = rotate(negative, angle, reshape=False, order=1, mode="reflect", prefilter=False)
    def normalize(value: np.ndarray) -> np.ndarray:
        value = value - np.quantile(value, 0.15)
        return np.clip(value / (np.quantile(value, 0.99) + 1e-8), 0.0, 1.0)
    positive = normalize(positive); negative = normalize(negative)
    image = _background(rng)
    target = np.zeros((SIZE, SIZE), dtype=bool)
    distractor = np.zeros_like(target)
    bank = np.zeros((8, SIZE, SIZE), dtype=np.float64)
    placements = [(rng.integers(8, 20), rng.integers(8, 20)), (rng.integers(58, 62), rng.integers(58, 62))]
    if rng.random() < 0.5:
        placements.reverse()
    for value, (y0, x0), is_target in ((positive, placements[0], True), (negative, placements[1], False)):
        height, width = value.shape
        image[y0 : y0 + height, x0 : x0 + width] += rng.uniform(0.65, 1.0) * value
        local = value >= max(0.34, float(np.quantile(value, 0.72)))
        if is_target:
            target[y0 : y0 + height, x0 : x0 + width] |= local
            mode = int(round((angle % 180) / 22.5)) % 8
            bank[mode, y0 : y0 + height, x0 : x0 + width] = np.maximum(bank[mode, y0 : y0 + height, x0 : x0 + width], gaussian_filter(local.astype(float), 0.8))
        else:
            distractor[y0 : y0 + height, x0 : x0 + width] |= local
    distractor &= ~target
    image = gaussian_filter(image, rng.uniform(0.25, 0.65)) + rng.normal(0.0, rng.uniform(0.02, 0.06), image.shape)
    sample = _package(image, target, distractor, bank, task, split, index)
    sample["mechanism_task"] = task
    return sample


def _package(image: np.ndarray, target: np.ndarray, distractor: np.ndarray, bank: np.ndarray, domain: str, split: str, index: int) -> dict[str, Any]:
    image = np.asarray(image, dtype=np.float64)
    image = (image - image.mean()) / (image.std() + 1e-8)
    gy, gx = np.gradient(image)
    channels = np.stack((image, gx, gy)).astype(np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    valid = (target_array > 0).astype(np.float32)
    return {
        "image": channels,
        "target": target_array,
        "distractor": np.asarray(distractor, dtype=np.float32),
        "orientation_bank": np.asarray(bank, dtype=np.float32),
        "orientation_valid": valid,
        "domain": domain,
        "split": split,
        "index": int(index),
    }


@lru_cache(maxsize=128)
def _generate(split: str, index: int) -> dict[str, Any]:
    rng = _rng(split, index)
    if split == "dev-natural" or (split == "train" and index % 2 == 0):
        return _natural(split, index, rng)
    return _mechanism(split, index, rng)


def generate_sample(split: str, index: int) -> dict[str, Any]:
    if split == "confirm":
        raise PermissionError("K2 confirm is hash-only until a later frozen phase")
    return _generate(split, index)


def _update_digest(digest: Any, sample: dict[str, Any]) -> None:
    for key in ("image", "target", "distractor", "orientation_bank", "orientation_valid"):
        digest.update(key.encode()); digest.update(np.asarray(sample[key]).tobytes())
    digest.update(str(sample["domain"]).encode())


def split_hash(split: str) -> str:
    digest = hashlib.sha256()
    for index in range(SPLIT_SIZES[split]):
        digest.update(index.to_bytes(4, "little")); _update_digest(digest, _generate(split, index))
    return digest.hexdigest()


def freeze_benchmark(result_root: Path) -> dict[str, Any]:
    manifest = {
        "version": VERSION,
        "size": SIZE,
        "split_sizes": SPLIT_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "train_mixture": "50% natural, 50% mechanism by fixed index parity",
        "mechanism_tasks": list(TASKS),
        "confirm_policy": "generated and content-hashed before model results; samples inaccessible",
        "hashes": {split: split_hash(split) for split in SPLIT_SIZES},
        "confirm_evaluated": False,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
