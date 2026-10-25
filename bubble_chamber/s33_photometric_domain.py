"""S33: source-independent photometric transforms for original AZConv/UNet.

All transforms are fit-free and act independently on a single source patch.
Training never computes source-006 statistics nor uses physically certified masks.
"""
from __future__ import annotations

import cv2
import numpy as np

MODES = ('raw', 'quantile', 'local')


def image_input(image: np.ndarray, mode: str = 'raw') -> np.ndarray:
    r"""Return float32 single-channel [0,1] with darkness as high response.

    Quantile: \(z=clip[(Q_{.9}(I)-I)/(Q_{.9}(I)-Q_{.1}(I)),0,1]\).
    Local: \(r=(G_5*I-I); z=clip[1/2+0.22*r/(Q_{.75}(|r|)+eps),0,1]\).

    Both cancel arbitrary constant image offsets. Quantile cancels positive
    intensity scaling when unclipped; local also cancels positive scaling.
    They need not be invariant to saturations, gamma, or sensor artifacts.
    """
    if mode not in MODES:
        raise ValueError(f'Unsupported S33 input normalization: {mode}')
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.dtype != np.uint8 or min(arr.shape) < 32:
        raise ValueError('Expected source-independent 2D uint8 H50Z crop or patch')
    gray = arr.astype(np.float32)
    if mode == 'raw':
        result = (255. - gray) / 255.
    elif mode == 'quantile':
        lo, hi = np.quantile(gray, [.10, .90])
        result = (hi - gray) / max(hi - lo, 1.)
    else:
        smooth = cv2.GaussianBlur(gray, (0,0), 5., borderType=cv2.BORDER_REFLECT_101)
        residual = smooth - gray
        scale = float(np.quantile(np.abs(residual), .75))
        result = .5 + (.22 / max(scale, .0001)) * residual
    return np.clip(result, 0., 1.).astype(np.float32)


def photometric_augment(image: np.ndarray, *, seed: int) -> np.ndarray:
    """One seeded independent affine exposure perturbation, no geometric warp.

    Avoids a biased data-dependent normalization fitted on held-out photo 006.
    """
    rng = np.random.default_rng(seed)
    contrast = float(rng.uniform(.65, 1.35))
    offset = float(rng.uniform(-38., 38.))
    return np.clip(image.astype(np.float32) * contrast + offset, 0, 255).astype(np.uint8)


def input_domain_audit(groups: dict[str, list[np.ndarray]]) -> dict:
    """Audit image statistics, not physical track segmentation quality."""
    if not groups or any(not len(v) for v in groups.values()):
        raise ValueError('Need nonempty source photo groups')
    out = {}
    for photo, imgs in sorted(groups.items()):
        means = [];medians=[];raw_q=[];local_q=[];quantile_q=[];highpass=[]
        for im in imgs:
            if im.ndim!=2 or im.dtype!=np.uint8:
                raise ValueError('Only verified source grayscale data')
            means.append(float(im.mean()));medians.append(float(np.median(im)))
            raw = image_input(im,'raw');loc = image_input(im,'local');qu=image_input(im,'quantile')
            raw_q.append(float(np.median(raw)));local_q.append(float(np.median(loc)));quantile_q.append(float(np.median(qu)))
            highpass.append(float(np.std(cv2.GaussianBlur(im.astype(np.float32),(0,0),5)-im.astype(np.float32))))
        out[photo]={'crops':len(imgs),'mean_gray':float(np.mean(means)),
                    'mean_of_crop_medians':float(np.mean(medians)),
                    'spread_of_crop_medians':float(np.std(medians)),
                    'mean_highpass_std':float(np.mean(highpass)),
                    'per_patch_median_of_raw_darkness':float(np.mean(raw_q)),
                    'per_patch_median_of_quantile_normalized':float(np.mean(quantile_q)),
                    'per_patch_median_of_local_normalized':float(np.mean(local_q))}
    return out
