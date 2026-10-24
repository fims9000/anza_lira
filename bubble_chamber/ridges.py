"""Experimental multi-scale dark-ridge detector for bubble-chamber photographs.

Frangi is a classical image-processing baseline, not an ANZA model. Its Hessian
ridge response is suitable for testing elongated low-contrast features, but
photographic marks, scratches and adjacent tracks can produce false responses.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as component_label
from skimage.filters import frangi


def frangi_dark_ridges(
    image_gray: np.ndarray,
    *,
    response_fraction: float = 0.12,
    scales: tuple[float, ...] = (1.0, 2.0, 3.0),
    min_area: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary foreground mask and 2D response in [0,1].

    Response threshold is a fraction of the maximum response of the *same*
    image. This is NOT a calibration between film runs. ``min_area`` applies
    to 8-connected foreground components; it does not ensure track identity.
    """
    image = np.asarray(image_gray)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 HxW")
    if not (0 < response_fraction < 1):
        raise ValueError("response_fraction must lie strictly between zero and one")
    if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("scales must be finite and positive")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    response = np.asarray(
        frangi(image.astype(np.float32) / 255.0, sigmas=scales, black_ridges=True),
        dtype=np.float32,
    )
    response[~np.isfinite(response)] = 0
    if response.max(initial=0.0) <= 0:
        return np.zeros_like(image, dtype=bool), response
    foreground = response >= response_fraction * float(response.max())
    components, _ = component_label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(components.ravel())
    foreground &= (components > 0) & (counts[components] >= min_area)
    return foreground, response
