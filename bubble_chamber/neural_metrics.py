"""Auditable binary-mask topology metrics for *synthetic* particle-track images.

These metrics cannot establish particle/track identity without instance labels.
Never interpret a small connected-component count as evidence of fewer false
bridges: merging two independent lines also reduces that count.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label
from skimage.morphology import skeletonize


_STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def topology_for_image(pred: np.ndarray, truth: np.ndarray, tolerance_px: int = 2) -> dict[str, float | int]:
    """Compute clDice and radius-tolerant centerline F1 on two binary masks.

    Standard clDice compares skeleton(pred) with mask(truth) and vice versa;
    tolerant F1 compares both skeletons after dilation by a disk radius.
    """
    p = np.asarray(pred, dtype=bool)
    y = np.asarray(truth, dtype=bool)
    if p.ndim != 2 or p.shape != y.shape:
        raise ValueError('pred and truth must be matching HxW masks')
    if not 0 <= tolerance_px <= 20:
        raise ValueError('tolerance_px must be in [0, 20]')
    if not p.any() and not y.any():
        return {'cldice': 1.0, 'centerline_f1': 1.0, 'centerline_precision': 1.0,
                'centerline_recall': 1.0, 'pred_components_8': 0, 'true_components_8': 0}
    sp = skeletonize(p)
    sy = skeletonize(y)
    if not sp.any() or not sy.any():
        cl = 0.0
        precision = recall = 0.0
    else:
        tprec = float(np.logical_and(sp, y).sum() / sp.sum())
        tsens = float(np.logical_and(sy, p).sum() / sy.sum())
        cl = 2 * tprec * tsens / max(tprec + tsens, 1e-12)
        yy, xx = np.ogrid[-tolerance_px:tolerance_px + 1, -tolerance_px:tolerance_px + 1]
        disk = xx * xx + yy * yy <= tolerance_px * tolerance_px
        precision = float(np.logical_and(sp, binary_dilation(sy, structure=disk)).sum() / sp.sum())
        recall = float(np.logical_and(sy, binary_dilation(sp, structure=disk)).sum() / sy.sum())
    centerline_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    n_pred = int(label(p, _STRUCTURE)[1])
    n_true = int(label(y, _STRUCTURE)[1])
    return {'cldice': cl, 'centerline_f1': centerline_f1,
            'centerline_precision': precision, 'centerline_recall': recall,
            'pred_components_8': n_pred, 'true_components_8': n_true}


def aggregate_topology(predictions: np.ndarray, truths: np.ndarray,
                       tolerance_px: int = 2) -> dict[str, float | int]:
    """Macro-average per image; explicit unit is image, NOT independent pixel."""
    preds = np.asarray(predictions)
    labels = np.asarray(truths)
    if preds.ndim != 3 or preds.shape != labels.shape or preds.shape[0] < 1:
        raise ValueError('expected equal nonempty [N,H,W] arrays')
    metrics = [topology_for_image(p, y, tolerance_px) for p, y in zip(preds, labels)]
    result: dict[str, float | int] = {'images': int(len(metrics)), 'tolerance_px': int(tolerance_px)}
    for name in ('cldice', 'centerline_f1', 'centerline_precision', 'centerline_recall',
                 'pred_components_8', 'true_components_8'):
        result[f'macro_{name}'] = float(np.mean([m[name] for m in metrics]))
    result['instance_link_accuracy_available'] = False
    return result
