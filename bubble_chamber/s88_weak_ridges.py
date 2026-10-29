"""S88: grayscale-only, scale-space dark-ridge candidates for bubble chamber film.

This is a Steger-inspired *candidate* detector, not a particle recognizer.
Steger-style transverse subpixel localization and Hessian orientation avoid
assuming that tracks are straight. The returned binary mask is NOT ground truth;
all pixels without a response remain physically UNKNOWN as well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (binary_propagation, gaussian_filter,
                           label as component_label, map_coordinates)


@dataclass(frozen=True)
class WeakRidgeConfig:
    sigmas: tuple[float, ...] = (1.0, 1.6, 2.4)
    low_snr: float = 1.7
    high_snr: float = 3.4
    max_tangent_curvature_ratio: float = 0.55
    max_subpixel_offset: float = 0.70
    min_component_pixels: int = 8
    min_dark_deficit: float = 1.0

    def validate(self) -> None:
        if (not self.sigmas or len(self.sigmas) > 8
                or any(not np.isfinite(s) or not 0.65 <= s <= 6 for s in self.sigmas)
                or tuple(sorted(self.sigmas)) != self.sigmas):
            raise ValueError('sigmas must be sorted finite scales in 0.65..6')
        if not (np.isfinite(self.low_snr) and np.isfinite(self.high_snr)
                and 0 < self.low_snr < self.high_snr <= 100):
            raise ValueError('require 0 < low_snr < high_snr <= 100')
        if not (0 <= self.max_tangent_curvature_ratio < 1):
            raise ValueError('max_tangent_curvature_ratio must be in [0,1)')
        if not (0.2 <= self.max_subpixel_offset <= 0.95):
            raise ValueError('max_subpixel_offset must be in 0.2..0.95')
        if self.min_component_pixels < 1 or not np.isfinite(self.min_dark_deficit) or self.min_dark_deficit < 0:
            raise ValueError('invalid minimum component size or dark deficit')


def _noise_scale(gray: np.ndarray) -> float:
    """Robust in-tile film-grain proxy; NOT a calibrated physical noise model."""
    resid = gray - gaussian_filter(gray, 1.25, mode='reflect')
    median = float(np.median(resid))
    # Floor prevents zero-noise drawings from generating infinite responses.
    return max(1.0, 1.4826 * float(np.median(np.abs(resid - median))))


def ridge_mask_from_score(score: np.ndarray, cfg: WeakRidgeConfig) -> np.ndarray:
    """Cheap threshold-only stage, reusable across a CAL threshold grid.

    Works on the image-only score before any physical interpretation.
    """
    cfg.validate()
    if score.ndim != 2 or not np.issubdtype(score.dtype, np.number):
        raise ValueError('expected numeric HxW ridge score')
    if not np.isfinite(score).all() or np.min(score) < 0:
        raise ValueError('score must be finite and nonnegative')
    weak = score >= cfg.low_snr
    strong = score >= cfg.high_snr
    mask = binary_propagation(strong, structure=np.ones((3, 3), dtype=np.uint8), mask=weak)
    components, _ = component_label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())
    mask &= (components > 0) & (sizes[components] >= cfg.min_component_pixels)
    return mask.astype(bool)


def dark_ridge_candidates(
    gray: np.ndarray, cfg: WeakRidgeConfig = WeakRidgeConfig(),
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
    """Return mask and per-pixel diagnostics (all UNKNOWN, not probabilities).

    At each Gaussian sigma compute the Hessian's largest positive eigenvalue,
    corresponding transverse normal, tangent curvature and first-order
    subpixel ridge offset t=-grad_normal/lambda_normal. Suppress non-maxima
    along the normal, retain anisotropic *dark* ridges, and apply weak/strong
    hysteresis. The maps measure image response only; not dE/dx, mass or ID.
    """
    cfg.validate()
    a = np.asarray(gray)
    if a.ndim != 2 or a.dtype != np.uint8 or min(a.shape) < 8:
        raise ValueError('expected uint8 HxW image, both dimensions >= 8')
    image = a.astype(np.float32)
    h, w = a.shape
    y, x = np.indices((h, w), dtype=np.float32)
    noise = _noise_scale(image)
    best_score = np.zeros_like(image)
    best_subpixel_xy = np.zeros((h, w, 2), dtype=np.float32)
    best_tangent_xy = np.zeros((h, w, 2), dtype=np.float32)
    best_scale = np.zeros_like(image)

    for sigma in cfg.sigmas:
        # Coordinates: d_x = derivative along image axis 1, d_y along axis 0.
        dx = gaussian_filter(image, sigma, order=(0, 1), mode='reflect')
        dy = gaussian_filter(image, sigma, order=(1, 0), mode='reflect')
        hxx = gaussian_filter(image, sigma, order=(0, 2), mode='reflect')
        hyy = gaussian_filter(image, sigma, order=(2, 0), mode='reflect')
        hxy = gaussian_filter(image, sigma, order=(1, 1), mode='reflect')
        root = np.hypot(hxx - hyy, 2 * hxy)
        normal_curvature = (hxx + hyy + root) * 0.5
        tangent_curvature = (hxx + hyy - root) * 0.5
        # Principal normal vector of the larger eigenvalue, safe at equal eigs.
        angle = 0.5 * np.arctan2(2 * hxy, hxx - hyy)
        nx, ny = np.cos(angle), np.sin(angle)
        signed_offset = -(dx * nx + dy * ny) / np.maximum(normal_curvature, 1e-8)

        response = np.maximum(0.0, (sigma * sigma / noise) * normal_curvature)
        # Line must be darker than its extended Gaussian neighborhood.
        local_background = gaussian_filter(image, max(2.8, 2.8 * sigma), mode='reflect')
        dark_deficit = local_background - image
        valid = ((normal_curvature > 1e-6)
                 & (np.abs(tangent_curvature) <= cfg.max_tangent_curvature_ratio * normal_curvature)
                 & (np.abs(signed_offset) <= cfg.max_subpixel_offset)
                 & (dark_deficit >= cfg.min_dark_deficit))
        # True 2D directional non-maximum suppression; adjacent plateaus may
        # remain but are never interpreted as physical track branches.
        before = map_coordinates(response, [y - ny, x - nx], order=1, mode='nearest')
        after = map_coordinates(response, [y + ny, x + nx], order=1, mode='nearest')
        valid &= (response >= before) & (response >= after)
        valid[:2, :] = False
        valid[-2:, :] = False
        valid[:, :2] = False
        valid[:, -2:] = False
        score = np.where(valid, response, 0.0)
        better = score > best_score
        best_score = np.maximum(best_score, score)
        best_scale[better] = sigma
        best_subpixel_xy[:, :, 0][better] = (x + signed_offset * nx)[better]
        best_subpixel_xy[:, :, 1][better] = (y + signed_offset * ny)[better]
        best_tangent_xy[:, :, 0][better] = -ny[better]
        best_tangent_xy[:, :, 1][better] = nx[better]

    mask = ridge_mask_from_score(best_score, cfg)
    return mask, {
        'score_snr_proxy': best_score,
        'selected_sigma_px': best_scale,
        'subpixel_xy_px': best_subpixel_xy,
        'tangent_xy': best_tangent_xy,
        'noise_sigma_proxy': noise,
        'semantic_status': 'MACHINE_CANDIDATES_ONLY_NOT_PHYSICAL_GT',
    }
