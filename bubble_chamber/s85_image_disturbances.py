"""Frozen, pixel-only artificial scratch/grain stress condition for S85.

Noise uses an explicit independent seed and NEVER consults track geometry or
physical labels. Does not claim real camera or physics-run variation.
"""
from __future__ import annotations

import cv2
import numpy as np

CONDITIONS = ('clean', 'blur_sigma_11', 'blur_sigma_16', 'scratch_grain')


def condition_pixels(gray: np.ndarray, condition: str, *, noise_seed: int):
    if not isinstance(gray, np.ndarray) or gray.shape != (192, 192) or gray.dtype != np.uint8:
        raise ValueError('Expected 192x192 uint8 image, no reference geometry')
    if condition == 'clean':
        return gray.copy()
    if condition in ('blur_sigma_11', 'blur_sigma_16'):
        sigma = 1.1 if condition == 'blur_sigma_11' else 1.6
        return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if condition != 'scratch_grain':
        raise ValueError('Only prerecorded S85 disturbances allowed')
    if type(noise_seed) is not int or noise_seed < 0:
        raise ValueError('Scratch and grain need an explicit nonnegative noise seed')
    rng = np.random.default_rng(noise_seed)
    image = gray.copy()
    for n in range(2):
        # Draw diagonal synthetic artefacts without avoiding injected tracks.
        x0 = int(rng.integers(0, 192))
        y0 = int(rng.integers(0, 192))
        length = int(rng.integers(30, 80))
        angle = float(rng.uniform(.40, 1.15))*(1 if n == 0 else -1)
        x1 = max(0, min(191, round(x0 + length*np.cos(angle))))
        y1 = max(0, min(191, round(y0 + length*np.sin(angle))))
        cv2.line(image, (x0, y0), (x1, y1), int(rng.integers(12, 67)), 1, cv2.LINE_AA)
    noise = rng.normal(0., 9., size=image.shape)
    return np.rint(np.clip(image.astype(np.float64)+noise, 0., 255.)).astype(np.uint8)
