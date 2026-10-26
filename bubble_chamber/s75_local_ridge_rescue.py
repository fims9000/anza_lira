"""S75 pixel-only selective ridge rescue; no labels, frame IDs or physics GT.

All parameters are frozen in S75_FROZEN_LOCAL_RIDGE_RESCUE_2026_10_26.md.
The output is an advisory image transformation, not an inferred physical track.
"""
from __future__ import annotations

import cv2
import numpy as np

ARMS = ('identity', 'coherent_dog_080', 'local_blackhat_070')


def _coherence(gray: np.ndarray) -> np.ndarray:
    """Local structure-tensor coherence [0,1], from image pixels only."""
    im = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), .9)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3) * .125
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3) * .125
    xx = cv2.GaussianBlur(gx * gx, (0, 0), 1.4)
    yy = cv2.GaussianBlur(gy * gy, (0, 0), 1.4)
    xy = cv2.GaussianBlur(gx * gy, (0, 0), 1.4)
    numerator = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy ** 2, 0.0))
    return np.clip(numerator / (xx + yy + 1e-4), 0.0, 1.0)


def preprocess(gray: np.ndarray, arm: str) -> np.ndarray:
    """Receive only 192x192 uint8 image; return independent uint8 image."""
    if not isinstance(gray, np.ndarray) or gray.shape != (192, 192) or gray.dtype != np.uint8:
        raise ValueError('S75 expects a uint8 image of shape (192,192)')
    if arm not in ARMS:
        raise ValueError('Unknown frozen S75 preprocessing arm')
    if arm == 'identity':
        return gray.copy()
    coherence = _coherence(gray)
    if arm == 'coherent_dog_080':
        fine = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), .65)
        coarse = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.2)
        dark_ridge = np.maximum(coarse - fine, 0.0)
        oriented = np.clip((coherence - .40) / .40, 0.0, 1.0)
        signal = np.clip((dark_ridge - 1.5) / 4.0, 0.0, 1.0)
        subtract = np.minimum(.8 * dark_ridge * oriented * signal, 20.0)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        signal = np.clip((response - 2.0) / 8.0, 0.0, 1.0)
        # Suppress isotropic grain and contrast amplification of uniform fields.
        gate = np.clip((coherence - .5) / .25, 0.0, 1.0)
        subtract = np.minimum(.70 * response * signal * gate, 16.0)
    return np.clip(np.rint(gray.astype(np.float32) - subtract), 0, 255).astype(np.uint8)


def choose_cal_only(cal: dict[str, dict[str, dict[str, int]]]) -> dict:
    """Selection uses synthetic CAL aggregate *only*, never TEST or image labels."""
    if set(cal) != set(ARMS):
        raise ValueError('Full S75 frozen CAL method grid required')
    conditions = ('clean', 'blur_sigma_11')
    metrics = ('selected_synthetic_correct_links', 'synthetic_noncorrect_selected',
               'complete_three_piece_paths_synthetic', 'solver_failure')
    for observations in cal.values():
        if set(observations) != set(conditions):
            raise ValueError('CAL must contain identical clean and blur conditions')
        for value in observations.values():
            if any(k not in value or type(value[k]) is not int or value[k] < 0 for k in metrics):
                raise ValueError('Invalid synthetic CAL metrics')
    clean0, blur0 = (cal['identity'][k] for k in conditions)
    eligible = []
    for arm in ARMS[1:]:
        c, b = (cal[arm][k] for k in conditions)
        if (b['selected_synthetic_correct_links'] >= blur0['selected_synthetic_correct_links'] + 2
            and b['synthetic_noncorrect_selected'] <= blur0['synthetic_noncorrect_selected']
            and b['complete_three_piece_paths_synthetic'] >= blur0['complete_three_piece_paths_synthetic']
            and c['selected_synthetic_correct_links'] >= clean0['selected_synthetic_correct_links'] - 2
            and c['synthetic_noncorrect_selected'] <= clean0['synthetic_noncorrect_selected'] + 2
            and c['complete_three_piece_paths_synthetic'] >= clean0['complete_three_piece_paths_synthetic'] - 1
            and b['solver_failure'] == 0 and c['solver_failure'] == 0):
            eligible.append(arm)
    selected = min(eligible, key=lambda x: (
        -cal[x]['blur_sigma_11']['selected_synthetic_correct_links'],
        cal[x]['blur_sigma_11']['synthetic_noncorrect_selected'],
        cal[x]['clean']['synthetic_noncorrect_selected'], x)) if eligible else 'identity'
    return {'chosen_CAL_only': selected, 'eligible_arms': eligible,
            'default_remains_original_S72': selected == 'identity',
            'native_physics_auto_join_authorized': False}
