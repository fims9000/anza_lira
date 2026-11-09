"""S102: original-image coarse-to-fine line context for bubble-chamber previews.

This is a dark-line image-evidence operator, not a particle recognizer. It
supplements S88/S92 and must not replace source-locked S86 or expert labels.
No physical join, missing bubble pixel, or particle species is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class ContextConfig:
    # Detector works on reduced global context ONLY; native-resolution S88 is
    # still responsible for the observed, local image evidence.
    max_context_side: int = 640
    orientations: int = 12
    along_sigmas: tuple[float, ...] = (5.5, 11.0)
    across_sigmas: tuple[float, ...] = (1.0, 2.5)
    min_context_contrast: float = 2.0
    max_candidate_fraction: float = 0.22
    angular_tick_max_length_px: float = 42.0
    angular_tick_min_turn_deg: float = 70.0
    long_trace_min_length_px: float = 32.0
    short_context_min_length_px: float = 12.0

    def validate(self) -> None:
        if (type(self.max_context_side) is not int or not 96 <= self.max_context_side <= 4096
                or type(self.orientations) is not int or not 4 <= self.orientations <= 32):
            raise ValueError('invalid S102 context image budget or orientation count')
        for seq in (self.along_sigmas, self.across_sigmas):
            if not seq or len(seq) > 4 or any(not math.isfinite(s) or not 0.5 <= s <= 32 for s in seq):
                raise ValueError('invalid S102 filter scales')
        if (not 0 < self.min_context_contrast <= 255
                or not 0 < self.max_candidate_fraction < 1
                or not 2 <= self.angular_tick_max_length_px <= 1000
                or not 0 < self.angular_tick_min_turn_deg < 180
                or not 4 <= self.long_trace_min_length_px <= 4000
                or not 2 <= self.short_context_min_length_px < self.long_trace_min_length_px):
            raise ValueError('invalid S102 ranking parameters')


def _dark_line_kernel(angle_rad: float, along_sigma: float, across_sigma: float) -> np.ndarray:
    # Matched long dark-line profile vs wider band. Both bands have unit sum,
    # so flat illumination gives a zero response; negative responds to white.
    radius = int(math.ceil(2.8 * max(along_sigma, 2.8 * across_sigma)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    u = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    v = -xx * np.sin(angle_rad) + yy * np.cos(angle_rad)
    core = np.exp(-0.5 * ((u / along_sigma) ** 2 + (v / across_sigma) ** 2))
    band = np.exp(-0.5 * ((u / along_sigma) ** 2 + (v / (2.8 * across_sigma)) ** 2))
    core /= core.sum()
    band /= band.sum()
    kernel = band - core  # dark line -> positive intensity response
    return kernel.astype(np.float32)


def oriented_dark_line_context(gray: np.ndarray, config: ContextConfig = ContextConfig()) -> dict:
    """Return global orientation/strength from multiple elongated convolutions.

    Orientation is modulo pi. Strength uses original gray-value units (not a
    physical confidence or calibrated detection probability).  No downsampled
    pixels may be counted as actual observed source-mask pixels.
    """
    config.validate()
    arr = np.asarray(gray)
    if arr.ndim != 2 or arr.dtype != np.uint8 or min(arr.shape) < 32:
        raise ValueError('S102 requires nonempty 2D uint8 image >=32x32')
    h, w = arr.shape
    ratio = min(1.0, config.max_context_side / max(h, w))
    nw, nh = max(32, int(round(w * ratio))), max(32, int(round(h * ratio)))
    small = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    best = np.zeros((nh, nw), np.float32)
    angle_map = np.zeros((nh, nw), np.float32)
    scale_map = np.zeros((nh, nw), np.float32)
    for along in config.along_sigmas:
        for across in config.across_sigmas:
            for orientation_idx in range(config.orientations):
                theta = math.pi * orientation_idx / config.orientations
                kernel = _dark_line_kernel(theta, along, across)
                value = cv2.filter2D(small, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)
                np.maximum(value, 0, out=value)
                better = value > best
                best[better] = value[better]
                angle_map[better] = theta
                scale_map[better] = across / ratio
    # Keep coarse evidence as a continuous map, not a hard global gate: thin
    # observed tracks can be lost by downsampling and must remain recoverable.
    strength = cv2.resize(best, (w, h), interpolation=cv2.INTER_LINEAR)
    orientation = cv2.resize(angle_map, (w, h), interpolation=cv2.INTER_NEAREST)
    scale = cv2.resize(scale_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return {
        'strength_gray_units': strength.astype(np.float32),
        'tangent_orientation_rad': orientation.astype(np.float32),
        'preferred_width_scale_px': scale.astype(np.float32),
        'context_downsample_ratio': float(ratio),
        'semantics': 'GLOBAL_IMAGE_LINE_EVIDENCE_NOT_PHYSICAL_PROBABILITY',
    }


def supplemental_candidate_mask(gray: np.ndarray, context: dict,
                                config: ContextConfig = ContextConfig()) -> tuple[np.ndarray, dict]:
    """Optional high-recall supplemental dark pixels; never treated as ground truth.

    A density guard refuses to turn textured/noisy regions into enormous
    foreground masks. The mask is intentionally independent of S88 so its
    incremental detections can be measured in a frozen ablation.
    """
    config.validate()
    image = np.asarray(gray)
    strength = np.asarray(context['strength_gray_units'])
    if image.ndim != 2 or image.dtype != np.uint8 or strength.shape != image.shape:
        raise ValueError('S102 context/source shape mismatch')
    local_bg = gaussian_filter(image.astype(np.float32), sigma=12, mode='reflect')
    dark_excess = local_bg - image.astype(np.float32)
    # An oriented response alone can be a long scratch: rank actual traces
    # downstream and never present this as an accepted particle mask.
    mask = (strength >= config.min_context_contrast) & (dark_excess >= 0.5)
    mask[:3, :] = False
    mask[-3:, :] = False
    mask[:, :3] = False
    mask[:, -3:] = False
    fraction = float(mask.mean())
    if fraction > config.max_candidate_fraction:
        return np.zeros_like(mask), {
            'status': 'DENSE_CONTEXT_ABSTAIN', 'raw_fraction': fraction,
            'source_pixels_accepted_as_physical': False,
        }
    return mask, {
        'status': 'SUPPLEMENTAL_VISUAL_CANDIDATES', 'raw_fraction': fraction,
        'source_pixels_accepted_as_physical': False,
    }


def _geometry(points_xy: np.ndarray) -> tuple[float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('S102 requires ordered 2D source points')
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    valid = lengths > 1e-8
    total = float(lengths.sum())
    dirs = steps[valid]
    if len(dirs) < 2:
        return total, 0.0
    angles = np.unwrap(np.arctan2(dirs[:, 1], dirs[:, 0]))
    changes = np.diff(angles)
    window = min(4, len(changes))
    angular = np.convolve(np.degrees(changes), np.ones(window), mode='valid')
    return total, float(np.max(np.abs(angular))) if len(angular) else 0.0


def rank_observed_fragments(components: list[dict], context: dict,
                            config: ContextConfig = ContextConfig()) -> list[dict]:
    """Group-independent VISUAL RANKING; never removes S86 source fragments.

    This output is intended for a clean one-color track overlay. Short angular
    glyphs are hidden in that optional overlay, retained for audit/review.
    Without source-matched independent ground truth this is not a classifier.
    """
    config.validate()
    support = np.asarray(context['strength_gray_units'])
    if support.ndim != 2 or not np.isfinite(support).all():
        raise ValueError('S102 invalid global context')
    h, w = support.shape
    seen = set()
    result = []
    for component in components:
        cid = component['component_id']
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('S102 source fragment identity duplicated or missing')
        seen.add(cid)
        xy = np.asarray(component['points_xy'], dtype=np.float64)
        length, sharp = _geometry(xy)
        x = np.rint(xy[:, 0]).astype(int)
        y = np.rint(xy[:, 1]).astype(int)
        if np.any((x < 0) | (x >= w) | (y < 0) | (y >= h)):
            raise ValueError('S102 source coordinates outside image')
        values = support[y, x]
        median_support = float(np.median(values))
        angular_tick = (length <= config.angular_tick_max_length_px and
                        sharp >= config.angular_tick_min_turn_deg)
        if angular_tick:
            status = 'HIDDEN_ANGULAR_GLYPH_REVIEW'
        elif length >= config.long_trace_min_length_px:
            status = 'VISIBLE_LONG_TRACE_CANDIDATE'
        elif (length >= config.short_context_min_length_px and
              median_support >= config.min_context_contrast):
            status = 'VISIBLE_CONTEXT_SUPPORTED_SHORT_CANDIDATE'
        else:
            status = 'HIDDEN_UNCERTAIN_SHORT_REVIEW'
        result.append({
            'component_id': cid,
            'preview_status': status,
            'display_in_clean_overlay': status.startswith('VISIBLE_'),
            'arc_length_px': round(length, 3),
            'sharp_local_turn_deg': round(sharp, 3),
            'global_dark_line_support_gray_units': round(median_support, 4),
            'physical_particle_identity': 'UNKNOWN',
            'source_observations_immutable': True,
        })
    return result


def visual_group_ids(source_ids: list[str], proposals: list[dict]) -> dict[str, str]:
    """Stable per-preview visual GROUP IDs; selected joins are unverified.

    Input proposals must be the already-selected, topology-safe S92/S101 set.
    A separate rendering layer must draw inferred gaps dashed (never solid).
    """
    if len(set(source_ids)) != len(source_ids) or any(not x for x in source_ids):
        raise ValueError('invalid unique source component IDs')
    parent = {x: x for x in source_ids}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    used = set()
    for p in proposals:
        if not p.get('inferred_only', False) or p.get('physical_auto_join', True):
            raise ValueError('S102 joins must remain hypothetical')
        a, b = p['component_a'], p['component_b']
        pa, pb = p['port_a'], p['port_b']
        if a not in parent or b not in parent or a == b or pa not in (0, 1) or pb not in (0, 1):
            raise ValueError('S102 invalid fragment link')
        if (a, pa) in used or (b, pb) in used or root(a) == root(b):
            raise ValueError('S102 conflicting, cyclic or repeated connection')
        used.update(((a, pa), (b, pb)))
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for cid in source_ids:
        groups.setdefault(root(cid), []).append(cid)
    lut = {}
    for members in groups.values():
        visual_id = 's102_visual_' + sha256('|'.join(sorted(members)).encode('utf-8')).hexdigest()[:16]
        lut.update({cid: visual_id for cid in members})
    return lut