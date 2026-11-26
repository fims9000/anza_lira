"""Train-only approximate fixed-LPS augmentation of cached 3D patches."""
import numpy as np
from scipy.ndimage import shift


def jitter_batch(x, targets, *, degree, voxel_mm, endpoint_sigma_mm, noise_mm, rng):
    """Recenter CT and dense target together; independently move individual tips.

    Uses a cached patch, not a new crop of the original CT: border extension
    is approximate. Reject shifts that move a tip outside the cached FOV.
    Only call on fitted training rows; validation/inference remain unchanged.
    """
    if x.ndim != 5 or x.shape[1] != 4 or degree not in (2, 3):
        raise ValueError('Expected separate-tip 3D patches')
    if targets.shape != (len(x), 1, *x.shape[2:]):
        raise ValueError('Dense target shape mismatch')
    if not all(np.isfinite(v) and v > 0 for v in (voxel_mm, endpoint_sigma_mm, noise_mm)):
        raise ValueError('Invalid augmentation scales')
    out = x.copy()
    dense = targets.copy()
    shape = np.array(x.shape[2:])
    grid = np.indices(tuple(shape), dtype=np.float32)
    sigma = endpoint_sigma_mm / voxel_mm
    applied = 0
    for i in range(len(x)):
        if rng.random() >= 0.5:
            continue
        centers = []
        for heat in x[i, 1:1+degree]:
            peak = np.array(np.unravel_index(np.argmax(heat), heat.shape))
            value = float(heat[tuple(peak)])
            if value <= 0 or not np.isfinite(value):
                raise ValueError('Missing endpoint heatmap')
            center = peak.astype(float)
            for axis in range(3):
                q = peak.copy()
                step = 1 if q[axis] < shape[axis]-1 else -1
                q[axis] += step
                neighbor = float(heat[tuple(q)])
                if neighbor <= 0:
                    raise ValueError('Invalid Gaussian endpoint neighborhood')
                center[axis] += step/2 + sigma**2 * np.log(neighbor/value)/step
            centers.append(center)
        centers = np.asarray(centers)
        delta = rng.normal(0, noise_mm / voxel_mm, size=centers.shape)
        mean_delta = delta.mean(axis=0)
        moved = centers + delta - mean_delta
        if np.any(moved < 0) or np.any(moved > shape-1):
            continue
        out[i, 0] = shift(x[i, 0], -mean_delta, order=1, mode='nearest', prefilter=False)
        dense[i, 0] = shift(targets[i, 0], -mean_delta, order=0, mode='nearest', prefilter=False)
        for j, center in enumerate(moved):
            diff = grid - center[:, None, None, None]
            out[i, j+1] = np.exp(-np.sum(diff*diff, axis=0)/(2*sigma*sigma))
        if degree == 2:
            out[i, 3] = 0
        applied += 1
    return out, dense, applied
